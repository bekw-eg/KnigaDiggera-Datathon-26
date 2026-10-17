"""Streaming ML orchestration and a Windows-safe wall-clock deadline supervisor."""

import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.schemas import CandidateEvent
from app.vision.config import VisionConfig
from app.vision.detector import YOLOAdapter
from app.vision.motion import MotionDetector, merge_intervals
from app.vision.reflection import ReflectionGuard
from app.vision.result import Coverage, Diagnostic, EvidenceFrame, VisionAnalysisResult
from app.vision.roi import load_rois
from app.vision.video import VideoError, VideoReader, encode_jpeg
from app.vision.vlm import OllamaVisionClient, VLMError, VLMObservation

logger = logging.getLogger(__name__)


def representative_frames(
    frames: list[EvidenceFrame],
    start: float,
    end: float,
    limit: int,
) -> list[EvidenceFrame]:
    """Chronological first/middle/last samples within a candidate; bounded to 1–3."""
    within = [frame for frame in frames if start <= frame.timestamp_sec <= end]
    if not within:
        return []
    count = min(limit, len(within))
    indices = [len(within) // 2] if count == 1 else np.linspace(0, len(within) - 1, count).round()
    return [within[int(index)] for index in indices]


def _analyze(
    video_path: Path,
    config: VisionConfig,
    deadline: float,
    progress: Callable[[VisionAnalysisResult], None] | None = None,
    client: OllamaVisionClient | None = None,
) -> VisionAnalysisResult:
    """Core implementation; internal client injection supports deterministic unit tests."""
    started = time.perf_counter()
    result = VisionAnalysisResult(coverage=Coverage(sample_fps=config.sample_fps))
    last_progress = 0.0

    def publish(force: bool = False) -> None:
        nonlocal last_progress
        result.metrics.total_sec = time.perf_counter() - started
        if progress and (force or time.monotonic() - last_progress > 0.5):
            progress(result)
            last_progress = time.monotonic()

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError

    def diagnose(code: str, severity: str = "warning", event_id: str | None = None) -> None:
        result.diagnostics.append(Diagnostic(code=code, severity=severity, event_id=event_id))
        logger.warning("vision_diagnostic code=%s", code)

    try:
        check_deadline()
        if not Path(video_path).is_file():
            raise VideoError("video_file_missing")
        run_dir = config.output_dir.resolve() / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        result.output_dir = str(run_dir)
        rois = load_rois(config.roi_path).active(config.camera_profile) if config.roi_path else []
        if not rois:
            diagnose("roi_classification_unavailable")
        motion = MotionDetector(config.motion, rois)
        hits = []
        opening = time.perf_counter()
        with VideoReader(video_path) as reader:
            result.metrics.decode_sec += time.perf_counter() - opening
            result.metadata = reader.metadata
            publish(True)
            frames = iter(reader.sample(fps=config.sample_fps, max_side=config.max_frame_side))
            while True:
                check_deadline()
                decoding = time.perf_counter()
                try:
                    frame = next(frames)
                except StopIteration:
                    result.metrics.decode_sec += time.perf_counter() - decoding
                    result.coverage.decoded_complete = True
                    result.coverage.decoded_fraction = 1.0
                    break
                result.metrics.decode_sec += time.perf_counter() - decoding
                result.metrics.decoded_frames = reader.decoded_frames
                result.coverage.decoded_until_sec = reader.decoded_until_sec
                duration = result.metadata.duration_sec
                if duration:
                    result.coverage.decoded_fraction = min(1, frame.timestamp_sec / duration)
                check_deadline()
                if result.metrics.sampled_frames >= config.max_sampled_frames:
                    diagnose("sample_frame_limit")
                    break
                cv_started = time.perf_counter()
                hits.extend(motion.update(frame.image, frame.timestamp_sec))
                result.metrics.cv_sec += time.perf_counter() - cv_started
                io_started = time.perf_counter()
                path = run_dir / f"frame_{result.metrics.sampled_frames:06d}.jpg"
                path.write_bytes(encode_jpeg(frame.image, config.jpeg_quality))
                result.evidence_frames.append(
                    EvidenceFrame(
                        path=str(path),
                        timestamp_sec=frame.timestamp_sec,
                        source_pts=frame.source_pts,
                        time_base=frame.time_base,
                        source_time_sec=frame.source_time_sec,
                    )
                )
                result.metrics.evidence_io_sec += time.perf_counter() - io_started
                result.metrics.sampled_frames += 1
                publish()
            result.metrics.decoded_frames = reader.decoded_frames
            result.coverage.decoded_until_sec = reader.decoded_until_sec
        check_deadline()
        intervals = merge_intervals(hits, config.motion, result.coverage.decoded_until_sec or 0)
        result.metrics.candidates_total = len(intervals)
        if len(intervals) > config.max_candidates:
            diagnose("candidate_limit")
        unknown = VLMObservation.unknown().tri_state()
        selections = []
        for interval in intervals[: config.max_candidates]:
            selected = representative_frames(
                result.evidence_frames, interval.start_sec, interval.end_sec, config.vlm.max_images
            )
            selections.append(selected)
            result.events.append(
                CandidateEvent(
                    event_id=uuid4().hex,
                    event_type="motion_activity",
                    start_sec=interval.start_sec,
                    end_sec=interval.end_sec,
                    frame_paths=[f.path for f in selected],
                    roi_id=interval.roi_id,
                    observations=unknown.copy(),
                )
            )
        publish(True)
        vlm = client or OllamaVisionClient(config.vlm)
        detector = YOLOAdapter(config.yolo_weights, config.yolo_enabled, config.yolo_device)
        roi_by_id = {roi.roi_id: roi for roi in rois}
        guard = ReflectionGuard()
        unavailable = False
        for event, selected in zip(result.events, selections, strict=True):
            check_deadline()
            if not selected:
                diagnose("candidate_has_no_frames", "error", event.event_id)
                continue
            if unavailable:
                diagnose("vlm_unavailable_for_candidate", "error", event.event_id)
                continue
            if result.metrics.vlm_calls >= config.max_vlm_calls:
                diagnose("vlm_call_limit", event_id=event.event_id)
                continue
            inference_started = time.perf_counter()
            result.metrics.vlm_calls += 1
            # Bound each call by the remaining deadline; supervisor handles stuck native I/O.
            vlm.config = config.vlm.model_copy(
                update={
                    "timeout_sec": max(
                        0.001, min(config.vlm.timeout_sec, deadline - time.monotonic())
                    )
                }
            )
            try:
                response = vlm.observe(
                    [Path(f.path) for f in selected],
                    [f.timestamp_sec for f in selected],
                    roi=roi_by_id.get(event.roi_id),
                )
                check_deadline()
                result.metrics.vlm_successes += 1
                result.coverage.vlm_intervals.append((event.start_sec, event.end_sec))
            except VLMError as exc:
                diagnose(exc.code, "error", event.event_id)
                unavailable = exc.code in {
                    "ollama_unavailable",
                    "model_missing",
                    "model_has_no_vision_capability",
                    "remote_model_forbidden",
                }
                continue
            finally:
                result.metrics.vlm_sec += time.perf_counter() - inference_started
            cv_started = time.perf_counter()
            detections = []
            height, width = frame.image.shape[:2]
            if config.yolo_enabled:
                try:
                    for evidence in selected:
                        image = cv2.imdecode(
                            np.frombuffer(Path(evidence.path).read_bytes(), np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                        detections.extend(detector.detect(image, rois))
                except (RuntimeError, ImportError):
                    diagnose("yolo_unavailable", "error", event.event_id)
            observation = guard.apply(
                response.observation,
                detections,
                roi_by_id.get("glass_partition"),
                width,
                height,
                region_is_glass=event.roi_id == "glass_partition",
            )
            event.observations = observation.tri_state()
            result.metrics.cv_sec += time.perf_counter() - cv_started
            publish(True)
        incomplete_codes = {"sample_frame_limit", "candidate_limit", "vlm_call_limit"}
        incomplete = not result.coverage.decoded_complete or any(
            d.severity == "error" or d.code in incomplete_codes for d in result.diagnostics
        )
        result.status = "partial" if incomplete else "complete"
    except TimeoutError:
        result.status = "timed_out"
        diagnose("processing_deadline", "error")
    except VideoError as exc:
        result.status = "partial" if result.metrics.sampled_frames else "failed"
        diagnose(str(exc), "error")
    except Exception as exc:
        # Avoid raw decoder, model and validation exception messages containing local data.
        result.status = "partial" if result.metrics.sampled_frames else "failed"
        diagnose(f"pipeline_{type(exc).__name__}", "error")
    publish(True)
    return result


def _worker(video: Path, config: VisionConfig, deadline: float, connection: Connection) -> None:
    try:
        result = _analyze(
            video,
            config,
            deadline,
            progress=lambda result: connection.send(
                {"kind": "progress", "result": result.model_dump(mode="json")}
            ),
        )
        connection.send({"kind": "result", "result": result.model_dump(mode="json")})
    finally:
        connection.close()


def analyze_video(video_path: Path, config: VisionConfig) -> VisionAnalysisResult:
    """Analyze locally in a supervised process; retain partial checkpoints on timeout.

    Synchronous entry point for a backend worker. On Windows, call from an importable
    module; CLI callers must use an if __name__ == '__main__' guard. Worker must allow
    child processes. Evidence files belong to caller and survive partial completion.
    """
    started = time.monotonic()
    deadline = started + config.max_seconds
    result = VisionAnalysisResult(coverage=Coverage(sample_fps=config.sample_fps))
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(Path(video_path), config, deadline, sender))
    terminal = False
    try:
        process.start()
        sender.close()
        while time.monotonic() < deadline:
            if receiver.poll(min(0.1, max(0, deadline - time.monotonic()))):
                try:
                    message = receiver.recv()
                    result = VisionAnalysisResult.model_validate(message["result"])
                    if message["kind"] == "result":
                        terminal = True
                        break
                except (EOFError, BrokenPipeError):
                    break
            elif not process.is_alive():
                break
        if not terminal and time.monotonic() >= deadline:
            result.status = "timed_out"
            result.diagnostics.append(Diagnostic(code="processing_deadline", severity="error"))
        elif not terminal:
            result.status = "partial" if result.metrics.sampled_frames else "failed"
            result.diagnostics.append(Diagnostic(code="worker_failed", severity="error"))
    except Exception as exc:
        result.status = "failed"
        result.diagnostics.append(
            Diagnostic(code=f"worker_start_or_ipc_{type(exc).__name__}", severity="error")
        )
    finally:
        sender.close()
        receiver.close()
        if process.is_alive():
            process.terminate()
        if process.pid is not None:
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            process.close()
    result.metrics.total_sec = time.monotonic() - started
    return result
