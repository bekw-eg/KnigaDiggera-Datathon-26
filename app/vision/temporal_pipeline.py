"""Bounded two-pass orchestration; the public deadline supervisor is shared."""

import logging
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.schemas import CandidateEvent, Observation
from app.vision.config import VisionConfig
from app.vision.desk import DeskMotionAnalyzer
from app.vision.detector import YOLOAdapter
from app.vision.evidence import (
    EventReplay,
    ReplayFrame,
    TemporalSelection,
    frame_fingerprint,
    prioritize_temporal_selections,
    select_temporal_frames,
)
from app.vision.reflection import ReflectionGuard
from app.vision.result import Coverage, Diagnostic, EvidenceFrame, VisionAnalysisResult
from app.vision.roi import load_rois
from app.vision.temporal import TemporalProposal, prioritize_and_deduplicate, propose_events
from app.vision.video import VideoError, VideoReader, encode_jpeg
from app.vision.vlm import ObjectKind, OllamaVisionClient, TemporalVLMObservation, VLMError

logger = logging.getLogger(__name__)


def analyze_temporal(
    video_path: Path,
    config: VisionConfig,
    deadline: float,
    progress: Callable[[VisionAnalysisResult], None] | None = None,
    client: OllamaVisionClient | None = None,
) -> VisionAnalysisResult:
    """Sequential overview/refinement, local evidence, and bounded local verification.

    Only refinement windows save pixels. Decoding is sequential from the origin on
    both passes to retain validation of every PTS. Long files can exhaust the decode
    budget before late windows; this is reported as incomplete coverage.
    """
    started = time.perf_counter()
    result = VisionAnalysisResult(
        coverage=Coverage(
            sample_fps=config.sample_fps,
            refine_fps=config.temporal.refine_fps,
        )
    )
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

    def checkpoint_events(proposals: list[TemporalProposal]) -> None:
        result.events = [
            CandidateEvent(
                event_id=uuid4().hex,
                event_type="motion_activity",
                start_sec=p.start_sec,
                end_sec=p.end_sec,
                frame_paths=[],
                roi_id=p.roi_id,
                observations=TemporalVLMObservation.unknown().tri_state(),
            )
            for p in proposals
        ]
        result.event_replay = [
            EventReplay(
                event_id=e.event_id,
                start_sec=p.start_sec,
                end_sec=p.end_sec,
                roi_id=p.roi_id,
                roi_ids=p.roi_ids,
                peak_sec=p.peak_sec,
                segmentation_insufficient=p.segmentation_insufficient,
                observations=e.observations.copy(),
            )
            for e, p in zip(result.events, proposals, strict=True)
        ]

    try:
        check_deadline()
        if not Path(video_path).is_file():
            raise VideoError("video_file_missing")
        run_dir = config.output_dir.resolve() / uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        result.output_dir = str(run_dir)
        rois = load_rois(config.roi_path).active(config.camera_profile) if config.roi_path else []
        desk_active = any(r.roi_id == config.temporal.desk_roi_id for r in rois)
        if not rois:
            diagnose("roi_classification_unavailable")
        if not desk_active:
            diagnose("desk_roi_unavailable_fallback")
        boundary = config.temporal.boundary
        if (
            boundary
            and boundary.enabled
            and (not desk_active or boundary.camera_profile != config.camera_profile)
        ):
            diagnose("boundary_calibration_unavailable")
            boundary = None
        motion = DeskMotionAnalyzer(config.motion, rois, config.temporal.desk_roi_id, boundary)
        scores = []
        with VideoReader(video_path) as reader:
            result.metadata = reader.metadata
            publish(True)
            frames = iter(
                reader.sample(
                    fps=config.sample_fps,
                    max_side=config.max_frame_side,
                    deadline=deadline,
                    max_decoded_frames=config.max_decoded_frames,
                )
            )
            try:
                while True:
                    check_deadline()
                    decoding = time.perf_counter()
                    try:
                        frame = next(frames)
                    except StopIteration:
                        result.coverage.decoded_complete = True
                        result.coverage.decoded_fraction = 1
                        break
                    finally:
                        result.metrics.decode_sec += time.perf_counter() - decoding
                    if result.metrics.sampled_frames >= config.max_sampled_frames:
                        diagnose("sample_frame_limit", "error")
                        break
                    cv_started = time.perf_counter()
                    scores.extend(motion.update(frame.image, frame.timestamp_sec))
                    result.metrics.cv_sec += time.perf_counter() - cv_started
                    result.metrics.sampled_frames += 1
                    result.coverage.decoded_until_sec = reader.decoded_until_sec
                    if reader.metadata.duration_sec:
                        result.coverage.decoded_fraction = min(
                            1, frame.timestamp_sec / reader.metadata.duration_sec
                        )
                    publish()
            finally:
                result.metrics.first_pass_decoded_frames = reader.decoded_frames
                result.metrics.decoded_frames = reader.decoded_frames
                result.coverage.decoded_until_sec = reader.decoded_until_sec
        check_deadline()
        cv_started = time.perf_counter()
        batch = propose_events(scores, config.temporal, result.coverage.decoded_until_sec or 0)
        proposals = prioritize_and_deduplicate(batch.proposals, rois, config.temporal)
        result.metrics.candidates_before_dedup = batch.total
        result.metrics.candidates_after_dedup = len(proposals)
        result.metrics.candidates_total = len(proposals)
        if batch.continuous_regions:
            diagnose("continuous_activity_undersegmented")
        if batch.truncated:
            diagnose("proposal_limit", "error")
        if len(proposals) > config.max_candidates:
            diagnose("candidate_limit", "error")
        proposals = proposals[: config.max_candidates]
        result.metrics.cv_sec += time.perf_counter() - cv_started
        checkpoint_events(proposals)
        publish(True)
        pool: list[ReplayFrame] = []
        refined_scores = []
        boundary_times: list[float] = []
        refined_until = 0.0
        if proposals:
            motion = DeskMotionAnalyzer(config.motion, rois, config.temporal.desk_roi_id, boundary)
            last_saved_sec = 0.0
            exhausted_windows = False
            with VideoReader(video_path) as reader:
                frames = iter(
                    reader.sample(
                        fps=config.temporal.refine_fps,
                        end_sec=max(p.end_sec for p in proposals),
                        max_side=config.max_frame_side,
                        deadline=deadline,
                        max_decoded_frames=max(
                            0, config.max_decoded_frames - result.metrics.first_pass_decoded_frames
                        ),
                    )
                )
                try:
                    while True:
                        check_deadline()
                        decoding = time.perf_counter()
                        try:
                            frame = next(frames)
                        except StopIteration:
                            exhausted_windows = True
                            break
                        finally:
                            result.metrics.decode_sec += time.perf_counter() - decoding
                        if not any(
                            p.start_sec <= frame.timestamp_sec <= p.end_sec for p in proposals
                        ):
                            motion.reset()
                            publish()
                            continue
                        if len(pool) >= config.temporal.max_refined_frames:
                            diagnose("refined_frame_limit", "error")
                            break
                        cv_started = time.perf_counter()
                        refined_scores.extend(motion.update(frame.image, frame.timestamp_sec))
                        signature = frame_fingerprint(frame.image)
                        if motion.boundary_activity:
                            boundary_times.append(frame.timestamp_sec)
                        result.metrics.cv_sec += time.perf_counter() - cv_started
                        io_started = time.perf_counter()
                        path = run_dir / f"refine_{len(pool):06d}.jpg"
                        path.write_bytes(encode_jpeg(frame.image, config.jpeg_quality))
                        evidence = EvidenceFrame(
                            path=str(path),
                            timestamp_sec=frame.timestamp_sec,
                            source_pts=frame.source_pts,
                            time_base=frame.time_base,
                            source_time_sec=frame.source_time_sec,
                        )
                        result.evidence_frames.append(evidence)
                        pool.append(
                            ReplayFrame(
                                **evidence.model_dump(), role="during", fingerprint=signature
                            )
                        )
                        result.metrics.second_pass_frames += 1
                        last_saved_sec = frame.timestamp_sec
                        result.metrics.evidence_io_sec += time.perf_counter() - io_started
                        publish()
                finally:
                    refined_until = (
                        (reader.decoded_until_sec or 0) if exhausted_windows else last_saved_sec
                    )
                    result.metrics.second_pass_decoded_frames = reader.decoded_frames
                    result.metrics.decoded_frames += reader.decoded_frames
            for p in proposals:
                if refined_until >= p.end_sec:
                    result.coverage.refined_intervals.append((p.start_sec, p.end_sec))
            cv_started = time.perf_counter()
            refined: list[TemporalProposal] = []
            refinement_config = config.temporal.refinement_config()
            for p in proposals:
                window_scores = [
                    s
                    for s in refined_scores
                    if s.start_sec >= p.start_sec
                    and s.end_sec <= p.end_sec
                    and s.roi_id == p.roi_id
                ]
                local = propose_events(window_scores, refinement_config, p.end_sec)
                if local.truncated:
                    diagnose("proposal_limit", "error")
                if not local.proposals:
                    diagnose("refinement_no_activity", "error")
                    refined.append(p.model_copy(update={"segmentation_insufficient": True}))
                for refined_proposal in local.proposals:
                    refined.append(
                        refined_proposal.model_copy(
                            update={
                                "start_sec": max(p.start_sec, refined_proposal.start_sec),
                                "end_sec": min(p.end_sec, refined_proposal.end_sec),
                                "roi_ids": p.roi_ids,
                            }
                        )
                    )
            refined = prioritize_and_deduplicate(refined, rois, config.temporal)
            result.metrics.refined_candidates = len(refined)
            if len(refined) > config.max_candidates:
                diagnose("candidate_limit", "error")
            prepared = prioritize_temporal_selections(
                [TemporalSelection(p, *select_temporal_frames(pool, p)) for p in refined]
            )[: config.max_candidates]
            proposals = [selection.proposal for selection in prepared]
            checkpoint_events(proposals)
            for index, (event, replay, selection) in enumerate(
                zip(result.events, result.event_replay, prepared, strict=True)
            ):
                p = selection.proposal
                selected, complete = selection.frames, selection.temporal_complete
                crop_roi = next((r for r in rois if r.roi_id == p.roi_id), None)
                selected = [
                    f.model_copy(
                        update={
                            "vlm_crop_roi_id": p.roi_id,
                            "vlm_crop_polygon": crop_roi.polygon if crop_roi else None,
                            "vlm_max_side": config.vlm.max_side,
                        }
                    )
                    for f in selected
                ]
                # Set together: validation must never see a transient incomplete triplet.
                replay_data = replay.model_dump()
                replay_data.update(
                    frames=selected,
                    temporal_complete=complete,
                    boundary_activity=any(p.start_sec <= t <= p.end_sec for t in boundary_times),
                )
                updated = EventReplay.model_validate(replay_data)
                result.event_replay[index] = updated
                event.frame_paths = [f.path for f in selected]
            result.metrics.cv_sec += time.perf_counter() - cv_started
            publish(True)
        vlm = client or OllamaVisionClient(config.vlm)
        detector = YOLOAdapter(config.yolo_weights, config.yolo_enabled, config.yolo_device)
        roi_by_id = {r.roi_id: r for r in rois}
        guard = ReflectionGuard()
        unavailable = False
        for event, replay in zip(result.events, result.event_replay, strict=True):
            check_deadline()
            if replay.segmentation_insufficient:
                replay.diagnostic_codes.append("temporal_segmentation_insufficient")
                diagnose("temporal_segmentation_insufficient", "error", event.event_id)
            if not replay.temporal_complete:
                replay.diagnostic_codes.append("temporal_evidence_incomplete")
                diagnose("temporal_evidence_incomplete", "error", event.event_id)
            if not replay.frames:
                replay.status = "incomplete"
                continue
            if unavailable:
                replay.status = "failed"
                replay.diagnostic_codes.append("vlm_unavailable_for_candidate")
                diagnose("vlm_unavailable_for_candidate", "error", event.event_id)
                continue
            if result.metrics.vlm_calls >= config.max_vlm_calls:
                replay.status = "skipped_budget"
                replay.diagnostic_codes.append("vlm_call_limit")
                diagnose("vlm_call_limit", "error", event.event_id)
                continue
            selected = (
                replay.frames
                if replay.temporal_complete and config.vlm.max_images == 3
                else [next((f for f in replay.frames if f.role == "during"), replay.frames[0])]
            )
            if len(selected) != 3 and replay.temporal_complete:
                replay.diagnostic_codes.append("temporal_vlm_image_budget")
                diagnose("temporal_vlm_image_budget", "error", event.event_id)
            inference_started = time.perf_counter()
            replay.vlm_frame_paths = [f.path for f in selected]
            result.metrics.vlm_calls += 1
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
                    temporal=True,
                    roles=[f.role for f in selected],
                )
                check_deadline()
                result.metrics.vlm_successes += 1
                result.coverage.vlm_intervals.append((event.start_sec, event.end_sec))
            except VLMError as exc:
                replay.status = "failed"
                replay.diagnostic_codes.append(exc.code)
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
                frame.image.shape[1],
                frame.image.shape[0],
                region_is_glass="glass_partition" in replay.roi_ids,
            )
            if not replay.temporal_complete or len(selected) != 3 or not desk_active:
                observation.object_transfer = Observation.UNKNOWN
                observation.transferred_object = ObjectKind.UNKNOWN
            event.observations = observation.tri_state()
            replay.observations = event.observations.copy()
            replay.status = (
                "verified" if replay.temporal_complete and len(selected) == 3 else "incomplete"
            )
            result.metrics.cv_sec += time.perf_counter() - cv_started
            publish(True)
        result.status = (
            "partial"
            if (
                not result.coverage.decoded_complete
                or any(d.severity == "error" for d in result.diagnostics)
            )
            else "complete"
        )
    except TimeoutError:
        result.status = "timed_out"
        diagnose("processing_deadline", "error")
    except VideoError as exc:
        result.status = (
            "timed_out"
            if str(exc) == "processing_deadline"
            else "partial"
            if result.metrics.sampled_frames
            else "failed"
        )
        diagnose(str(exc), "error")
    except Exception as exc:
        result.status = "partial" if result.metrics.sampled_frames else "failed"
        diagnose(f"pipeline_{type(exc).__name__}", "error")
    for replay in result.event_replay:
        if replay.status == "pending":
            replay.status = "incomplete"
            replay.diagnostic_codes.append("analysis_interrupted")
    publish(True)
    return result
