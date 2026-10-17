import time
from pathlib import Path

from conftest import make_video

from app.vision.config import VisionConfig
from app.vision.pipeline import _analyze, analyze_video
from app.vision.result import Coverage, VisionAnalysisResult
from app.vision.vlm import OllamaVisionClient


def run(video, config, sdk):
    return _analyze(
        video, config, time.monotonic() + 30, client=OllamaVisionClient(config.vlm, sdk)
    )


def test_motion_produces_serializable_candidate_and_exact_evidence(video, tmp_path, sdk):
    result = run(video, VisionConfig(output_dir=tmp_path / "out"), sdk)
    assert result.status == "complete"
    assert result.metrics.candidates_total == 1
    assert result.metrics.vlm_calls == result.metrics.vlm_successes == 1
    assert result.metrics.decoded_frames == 20 and result.metrics.sampled_frames == 4
    assert result.coverage.decoded_complete and result.coverage.decoded_fraction == 1
    assert len(result.events) == 1 and result.events[0].event_type == "motion_activity"
    by_path = {frame.path: frame for frame in result.evidence_frames}
    for path in result.events[0].frame_paths:
        assert Path(path).is_file()
        evidence = by_path[path]
        assert evidence.source_pts >= 0 and evidence.time_base
    assert '"state"' not in result.model_dump_json()
    assert all(value == "unknown" for value in result.events[0].observations.values())


def test_no_motion_no_vlm_and_no_candidate(tmp_path, sdk):
    video = make_video(tmp_path / "static.mov", moving=False)
    result = run(video, VisionConfig(output_dir=tmp_path / "out"), sdk)
    assert result.status == "complete" and not result.events and not sdk.calls


def test_vlm_failure_keeps_unknown_candidate_and_partial_status(video, tmp_path, sdk):
    sdk.error = ConnectionError()
    result = run(video, VisionConfig(output_dir=tmp_path / "out"), sdk)
    assert result.status == "partial" and len(result.events) == 1
    assert result.metrics.vlm_successes == 0
    assert any(d.code == "ollama_unavailable" for d in result.diagnostics)
    assert result.events[0].observations["cash_visible"] == "unknown"


def test_vlm_budget_is_not_successful_absence(video, tmp_path, sdk):
    result = run(video, VisionConfig(output_dir=tmp_path / "out", max_vlm_calls=0), sdk)
    assert result.status == "partial" and result.events
    assert any(d.code == "vlm_call_limit" for d in result.diagnostics)


def test_sample_limit_reports_partial_coverage(video, tmp_path, sdk):
    result = run(video, VisionConfig(output_dir=tmp_path / "out", max_sampled_frames=2), sdk)
    assert result.status == "partial" and not result.coverage.decoded_complete
    assert result.coverage.decoded_fraction < 1


def test_missing_video_and_invalid_roi_are_failures(tmp_path, sdk, video):
    result = run(tmp_path / "missing.mov", VisionConfig(output_dir=tmp_path / "out"), sdk)
    assert result.status == "failed" and any(d.severity == "error" for d in result.diagnostics)
    bad_roi = tmp_path / "bad.json"
    bad_roi.write_text("{}")
    result = run(video, VisionConfig(output_dir=tmp_path / "out", roi_path=bad_roi), sdk)
    assert result.status == "failed" and not result.events


def test_internal_deadline_is_explicit(video, tmp_path):
    result = _analyze(video, VisionConfig(output_dir=tmp_path / "out"), time.monotonic() - 1)
    assert result.status == "timed_out"


def test_public_windows_spawn_and_deadline(tmp_path):
    video = make_video(tmp_path / "static.mov", moving=False)
    result = analyze_video(video, VisionConfig(output_dir=tmp_path / "out", max_seconds=20))
    assert result.status == "complete" and result.metrics.sampled_frames == 4, (
        result.model_dump_json()
    )
    started = time.monotonic()
    timeout = analyze_video(video, VisionConfig(output_dir=tmp_path / "out", max_seconds=0.01))
    assert timeout.status == "timed_out"
    assert time.monotonic() - started < 3


def _stalled_worker(video, config, deadline, connection):
    result = VisionAnalysisResult(coverage=Coverage(sample_fps=config.sample_fps))
    result.metrics.sampled_frames = 7
    connection.send({"kind": "progress", "result": result.model_dump(mode="json")})
    time.sleep(30)


def test_supervisor_terminates_stalled_worker_and_preserves_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr("app.vision.pipeline._worker", _stalled_worker)
    started = time.monotonic()
    result = analyze_video(tmp_path / "unused.mov", VisionConfig(max_seconds=2))
    assert result.status == "timed_out"
    assert result.metrics.sampled_frames == 7
    assert time.monotonic() - started < 4
