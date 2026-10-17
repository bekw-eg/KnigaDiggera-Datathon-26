"""End-to-end generated MOVs with mock local Ollama."""

import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from app.schemas import CandidateEvent
from app.vision.config import VisionConfig
from app.vision.pipeline import _analyze, analyze_video
from app.vision.result import VisionAnalysisResult
from app.vision.roi import ROI, ROIConfig
from app.vision.temporal import TemporalConfig, TransferBoundary
from app.vision.vlm import OllamaVisionClient, VLMConfig


def scene(path, *, continuous=False, moving=True, outside=False, seconds=7):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        stream.time_base = Fraction(1, 1000)
        for index in range(seconds * 20):
            t = index / 20
            x = 10
            if moving:
                x = (
                    10 + index * 7 % 110
                    if continuous
                    else round(
                        10
                        + 65 * min(1, max(0, (t - 1.5) / 0.75))
                        - 45 * min(1, max(0, (t - 4.5) / 0.75))
                    )
                )
            image = np.zeros((120, 160, 3), np.uint8)
            y = 5 if outside else 50
            image[y : y + 18, x : x + 18] = 255
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index * 50, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def configuration(tmp_path, **kwargs):
    roi = tmp_path / "roi.local.json"
    polygon = [(0, 0.3), (1, 0.3), (1, 1), (0, 1)]
    roi.write_text(
        ROIConfig(
            camera_profile="test",
            calibrated=True,
            demonstration=False,
            rois=[
                ROI(roi_id=name, polygon=polygon) for name in ["desk", "cash_area", "transfer_zone"]
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    return VisionConfig(
        output_dir=tmp_path / "out",
        roi_path=roi,
        camera_profile="test",
        temporal=TemporalConfig(enabled=True),
        vlm=VLMConfig(max_images=3, num_ctx=4096),
        **kwargs,
    )


def run(path, config, sdk):
    sdk.payload.update(
        object_movement="unknown",
        hand_object_interaction="unknown",
        position_change_trackable="unknown",
        object_occluded="unknown",
    )
    return _analyze(path, config, time.monotonic() + 30, client=OllamaVisionClient(config.vlm, sdk))


def test_two_pass_localizes_actions_dedups_and_links_original_jpegs(tmp_path, sdk):
    path = scene(tmp_path / "actions.mov")
    config = configuration(tmp_path)
    result = run(path, config, sdk)
    assert result.status == "complete", result.model_dump_json()
    assert result.metrics.first_pass_decoded_frames == 140
    assert result.metrics.sampled_frames == 14 and result.metrics.second_pass_frames > 14
    assert result.metrics.decoded_frames == (
        result.metrics.first_pass_decoded_frames + result.metrics.second_pass_decoded_frames
    )
    assert result.metrics.candidates_before_dedup == 6
    assert result.metrics.candidates_after_dedup == 2
    assert len(result.events) == len(result.event_replay) == len(sdk.calls) == 2
    assert result.coverage.refined_intervals
    evidence = {f.path: f for f in result.evidence_frames}
    for event, replay, call in zip(result.events, result.event_replay, sdk.calls, strict=True):
        CandidateEvent.model_validate(event.model_dump())
        assert replay.event_id == event.event_id and replay.status == "verified"
        assert replay.temporal_complete and replay.roi_ids == ["cash_area", "desk", "transfer_zone"]
        assert [f.role for f in replay.frames] == ["before", "during", "after"]
        assert event.frame_paths == [f.path for f in replay.frames]
        assert replay.vlm_frame_paths == event.frame_paths
        assert all(f.vlm_crop_polygon and f.vlm_crop_roi_id == event.roi_id for f in replay.frames)
        assert len(call["messages"][1]["images"]) == 3
        assert call["options"]["num_ctx"] == 4096
        assert all(Path(f.path).read_bytes().startswith(b"\xff\xd8") for f in replay.frames)
        assert all(f.source_pts == evidence[f.path].source_pts for f in replay.frames)
        assert event.end_sec - event.start_sec <= 4
    assert VisionAnalysisResult.model_validate_json(result.model_dump_json()) == result
    assert not {"verdict", "rule_id", "violation"} & set(result.events[0].model_dump())


@pytest.mark.parametrize("outside,moving", [(False, False), (True, True)])
def test_static_and_outside_desk_do_not_call_vlm(tmp_path, sdk, outside, moving):
    result = run(
        scene(tmp_path / "quiet.mov", outside=outside, moving=moving), configuration(tmp_path), sdk
    )
    assert result.status == "complete" and not result.events and not sdk.calls


def test_continuous_movie_is_undersegmented_and_transfer_unknown(tmp_path, sdk):
    result = run(
        scene(tmp_path / "continuous.mov", continuous=True, seconds=12),
        configuration(tmp_path),
        sdk,
    )
    assert result.status == "partial", result.model_dump_json()
    assert len(result.events) > 1
    assert "continuous_activity_undersegmented" in {d.code for d in result.diagnostics}
    assert any(r.segmentation_insufficient for r in result.event_replay)
    for event, replay in zip(result.events, result.event_replay, strict=True):
        assert event.end_sec - event.start_sec <= 4
        if replay.segmentation_insufficient:
            assert (
                not replay.temporal_complete and event.observations["object_transfer"] == "unknown"
            )


def test_budget_caps_candidates_and_vlm_without_claiming_verification(tmp_path, sdk):
    path = scene(tmp_path / "actions.mov")
    result = run(path, configuration(tmp_path, max_candidates=1, max_vlm_calls=0), sdk)
    assert result.status == "partial" and len(result.events) == 1
    assert {"candidate_limit", "vlm_call_limit"} <= {d.code for d in result.diagnostics}
    assert result.event_replay[0].status == "skipped_budget" and not sdk.calls


def test_vlm_error_is_visible_in_replay_and_result(tmp_path, sdk):
    sdk.error = ConnectionError()
    result = run(scene(tmp_path / "actions.mov"), configuration(tmp_path), sdk)
    assert result.status == "partial" and result.metrics.vlm_successes == 0
    assert result.event_replay[0].status == "failed"
    assert "ollama_unavailable" in result.event_replay[0].diagnostic_codes
    assert all(e.observations["object_transfer"] == "unknown" for e in result.events)


def test_no_calibration_boundary_fallback_cannot_confirm_transfer(tmp_path, sdk):
    config = configuration(tmp_path)
    config.camera_profile = "other"
    config.temporal.boundary = TransferBoundary(
        enabled=True, camera_profile="test", points=((0.5, 0), (0.5, 1))
    )
    sdk.payload.update(object_transfer="yes", transferred_object="cash", cash_visible="yes")
    result = run(scene(tmp_path / "actions.mov"), config, sdk)
    assert {"desk_roi_unavailable_fallback", "boundary_calibration_unavailable"} <= {
        d.code for d in result.diagnostics
    }
    assert all(e.observations["object_transfer"] == "unknown" for e in result.events)


def test_refine_and_decode_limits_are_partial(tmp_path, sdk):
    path = scene(tmp_path / "actions.mov")
    config = configuration(tmp_path)
    config.temporal.max_refined_frames = 3
    result = run(path, config, sdk)
    assert result.status == "partial" and result.metrics.second_pass_frames == 3
    assert "refined_frame_limit" in {d.code for d in result.diagnostics}
    config = configuration(tmp_path, max_decoded_frames=150)
    result = run(path, config, sdk)
    assert result.status == "partial"
    assert "decode_frame_limit" in {d.code for d in result.diagnostics}
    assert result.metrics.decoded_frames <= 150


def test_expired_deadline_and_public_spawn(tmp_path, sdk):
    path = scene(tmp_path / "actions.mov")
    config = configuration(tmp_path)
    # Expired deadlines are visible; separate supervisor tests cover native stalls.
    result = _analyze(path, config, time.monotonic() - 1)
    assert result.status == "timed_out"
    quiet = scene(tmp_path / "quiet.mov", moving=False)
    result = analyze_video(quiet, config)
    assert result.status == "complete" and not result.events


def test_single_brief_action_and_incomplete_sequence(tmp_path, sdk):
    config = configuration(tmp_path)
    result = run(scene(tmp_path / "one.mov", seconds=4), config, sdk)
    assert result.status == "complete" and len(result.events) == 1
    assert result.event_replay[0].temporal_complete
    result = run(scene(tmp_path / "clipped.mov", seconds=2, continuous=True), config, sdk)
    assert result.status == "partial" and result.events
    assert all(not r.temporal_complete for r in result.event_replay)
    assert all(e.observations["object_transfer"] == "unknown" for e in result.events)


def test_deadline_during_second_pass_retains_pending_candidates(tmp_path, sdk, monkeypatch):
    from app.vision.video import VideoError, VideoReader

    original = VideoReader.sample

    def interrupted(self, *args, **kwargs):
        for frame in original(self, *args, **kwargs):
            if kwargs.get("fps") == 8 and frame.timestamp_sec > 1:
                raise VideoError("processing_deadline")
            yield frame

    monkeypatch.setattr(VideoReader, "sample", interrupted)
    result = run(scene(tmp_path / "actions.mov"), configuration(tmp_path), sdk)
    assert result.status == "timed_out" and result.events and not sdk.calls
    assert result.metrics.second_pass_decoded_frames > 0
    assert all(r.status == "incomplete" for r in result.event_replay)
    assert result.coverage.decoded_complete and not result.coverage.refined_intervals


def test_vfr_in_two_passes_keeps_actual_pts_and_brief_pulse_can_be_missed(tmp_path, sdk):
    from conftest import make_video

    from scripts.synthetic_temporal import make_temporal_scene

    config = configuration(tmp_path)
    path = make_video(tmp_path / "vfr.mov", times=[5000, 5100, 5400, 5900, 6300, 6800, 7000])
    result = run(path, config, sdk)
    assert result.status in {"complete", "partial"}
    assert result.evidence_frames
    assert not any(d.code.startswith("pipeline_") for d in result.diagnostics)
    assert all(
        abs(f.timestamp_sec - (f.source_time_sec - 5)) < 1e-8 for f in result.evidence_frames
    )
    pulse = make_temporal_scene(tmp_path / "pulse.mov")
    result = run(pulse, config, sdk)
    assert len(result.events) == 2
    assert all(not (e.start_sec <= 3.4 <= e.end_sec) for e in result.events)
    assert result.metrics.vlm_calls == 2


def test_vlm_overrun_is_not_reported_as_successful_coverage(tmp_path, sdk, monkeypatch):
    import app.vision.temporal_pipeline as pipeline

    config = configuration(tmp_path)
    sdk.payload.update(
        object_movement="unknown",
        hand_object_interaction="unknown",
        position_change_trackable="unknown",
        object_occluded="unknown",
    )
    client = OllamaVisionClient(config.vlm, sdk)
    original = client.observe
    deadline = time.monotonic() + 30

    def overrun(*args, **kwargs):
        response = original(*args, **kwargs)
        monkeypatch.setattr(pipeline.time, "monotonic", lambda: deadline + 1)
        return response

    monkeypatch.setattr(client, "observe", overrun)
    result = _analyze(scene(tmp_path / "actions.mov"), config, deadline, client=client)
    assert result.status == "timed_out" and result.metrics.vlm_calls == 1
    assert result.metrics.vlm_successes == 0 and not result.coverage.vlm_intervals


def test_glass_provenance_still_applies_reflection_guard_to_temporal_response(tmp_path, sdk):
    from app.vision.vlm import TemporalVLMObservation

    config = configuration(tmp_path)
    rois = ROIConfig.model_validate_json(config.roi_path.read_text(encoding="utf-8"))
    rois.rois.append(ROI(roi_id="glass_partition", polygon=rois.rois[0].polygon))
    config.roi_path.write_text(rois.model_dump_json(), encoding="utf-8")
    sdk.payload = TemporalVLMObservation.unknown().model_dump(mode="json")
    sdk.payload.update(
        people_visible="yes",
        object_transfer="yes",
        transferred_object="cash",
        cash_visible="yes",
        object_movement="yes",
        hand_object_interaction="yes",
        position_change_trackable="yes",
        object_occluded="no",
        reflection_possible="no",
    )
    result = _analyze(
        scene(tmp_path / "one.mov", seconds=4),
        config,
        time.monotonic() + 30,
        client=OllamaVisionClient(config.vlm, sdk),
    )
    assert result.status == "complete" and result.metrics.vlm_successes == 1
    assert "glass_partition" in result.event_replay[0].roi_ids
    assert result.events[0].observations["reflection_possible"] == "yes"
    assert result.events[0].observations["physical_presence"] == "unknown"
    assert result.events[0].observations["object_transfer"] == "unknown"


def test_previous_result_envelope_and_candidate_contract_remain_readable():
    previous = {
        "status": "complete",
        "coverage": {"sample_fps": 2},
        "events": [
            {
                "event_id": "legacy",
                "event_type": "motion_activity",
                "start_sec": 0,
                "end_sec": 1,
                "frame_paths": [],
                "roi_id": None,
                "observations": {"cash_visible": "unknown"},
            }
        ],
    }
    result = VisionAnalysisResult.model_validate(previous)
    assert result.event_replay == [] and result.metrics.second_pass_frames == 0
    assert result.events[0].model_dump(mode="json") == previous["events"][0]
