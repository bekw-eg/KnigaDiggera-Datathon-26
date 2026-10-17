import json
import os
import time

import cv2
import numpy as np
import pytest
from conftest import make_video
from pydantic import ValidationError

from app.vision.calibration import (
    CalibrationError,
    CalibrationSession,
    DisplayTransform,
    load_calibration_frame,
    save_rois,
)
from app.vision.config import VisionConfig
from app.vision.motion import MotionConfig, MotionDetector
from app.vision.pipeline import _analyze
from app.vision.roi import ROI, ROIConfig, load_rois
from app.vision.video import VideoError
from app.vision.vlm import OllamaVisionClient
from scripts.calibrate_roi import main
from scripts.roi_editor import ROIEditor, require_gui

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def drawn_session(**kwargs):
    session = CalibrationSession("camera-1", **kwargs)
    session.change(SQUARE)
    return session


@pytest.mark.parametrize("width,height,side", [(1920, 1080, 720), (97, 199, 160), (11, 2, 160)])
def test_display_normalization_and_round_trip(width, height, side):
    transform = DisplayTransform.fit(width, height, side)
    assert transform.width <= width and transform.height <= height
    assert transform.normalize(0, 0) == (0, 0)
    assert transform.normalize(transform.width - 1, transform.height - 1) == (1, 1)
    roi = ROI(roi_id="desk", polygon=SQUARE)
    assert [transform.pixel(p) for p in roi.polygon] == list(
        map(tuple, roi.pixels(transform.width, transform.height))
    )
    for x, y in [(0, 0), (transform.width // 2, transform.height // 2)]:
        assert transform.pixel(transform.normalize(x, y)) == (x, y)


@pytest.mark.parametrize("x,y", [(-1, 0), (100, 0), (0, 50), (float("nan"), 0), (0, float("inf"))])
def test_display_rejects_outside_or_nonfinite_points(x, y):
    with pytest.raises(ValueError):
        DisplayTransform(100, 50).normalize(x, y)


@pytest.mark.parametrize(
    "profile", ["", " ", " camera", "camera ", "camera\n", "../camera", "x" * 129]
)
def test_invalid_profiles_rejected_at_all_boundaries(profile):
    with pytest.raises(ValidationError):
        ROIConfig(camera_profile=profile, rois=[])
    with pytest.raises(ValidationError):
        VisionConfig(camera_profile=profile)
    with pytest.raises(ValidationError):
        CalibrationSession(profile)


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0), (1, 1)],
        [(0, 0), (float("inf"), 0), (0, 1)],
        [(0, 0), (1, 0), (0.5, 0), (0.5, 1), (0, 1)],
        [(0, 0), (1, 1), (0, 1), (1, 0.2)],  # Crossing with nonzero signed area.
        [(0, 0), (0.00001, 0), (0, 0.00001)],
        SQUARE + [(0, 0)],
        [(0, 0), (1, 0), (0, 1.01)],
    ],
)
def test_invalid_drafts_cannot_be_saved(tmp_path, points):
    session = CalibrationSession("camera-1")
    session.change(points)
    with pytest.raises(ValidationError):
        save_rois(tmp_path / "roi.json", session.configuration())
    assert not (tmp_path / "roi.json").exists()


def test_draft_review_confirmation_and_edit_invalidation(tmp_path):
    session = drawn_session()
    path = tmp_path / "камера.json"
    save_rois(path, session.configuration())
    assert not load_rois(path).calibrated and not load_rois(path).active("camera-1")
    with pytest.raises(CalibrationError, match="review"):
        session.configuration(confirm=True)
    session.reviewed = True
    save_rois(path, session.configuration(confirm=True), overwrite=True)
    loaded = load_rois(path)
    assert loaded.active("camera-1") and not loaded.active("camera-2")
    session.change([(0, 0), (0.5, 0), (0, 1)])
    assert not session.reviewed and not session.configuration().calibrated
    with pytest.raises(CalibrationError):
        session.configuration(confirm=True)


def test_demo_cannot_activate_including_loaded_demo(tmp_path):
    demo = drawn_session(demonstration=True)
    demo.reviewed = True
    with pytest.raises(CalibrationError):
        demo.configuration(confirm=True)
    path = tmp_path / "demo.json"
    save_rois(path, demo.configuration())
    loaded = load_rois(path)
    assert not loaded.active("camera-1")
    session = CalibrationSession("camera-1", loaded)
    session.reviewed = True
    with pytest.raises(CalibrationError):
        session.configuration(confirm=True)
    with pytest.raises(CalibrationError, match="mismatch"):
        CalibrationSession("camera-2", loaded)
    assert not load_rois("configs/roi.blank.example.json").active("YOUR_CAMERA_PROFILE")


def test_safe_zone_requires_explicit_visibility():
    session = drawn_session()
    with pytest.raises(CalibrationError, match="safe_visible"):
        session.select("safe_area")
    allowed = drawn_session(safe_visible=True)
    allowed.select("safe_area")
    allowed.change(SQUARE)
    config = allowed.configuration()
    assert len(config.rois) == 2
    with pytest.raises(CalibrationError, match="safe_visible"):
        CalibrationSession("camera-1", config).configuration()


def test_duplicate_ids_and_unknown_editor_zone_rejected():
    roi = ROI(roi_id="desk", polygon=SQUARE)
    with pytest.raises(ValidationError):
        ROIConfig(camera_profile="cam", rois=[roi, roi])
    with pytest.raises(CalibrationError, match="unsupported"):
        CalibrationSession(
            "cam", ROIConfig(camera_profile="cam", rois=[ROI(roi_id="unknown", polygon=SQUARE)])
        )


def test_atomic_save_failure_and_explicit_overwrite(tmp_path, monkeypatch):
    path = tmp_path / "out.json"
    draft = drawn_session().configuration()
    save_rois(path, draft)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_rois(path, draft)
    assert path.read_bytes() == before and not list(tmp_path.glob(".roi-*.tmp"))

    def failed_replace(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(os, "replace", failed_replace)
    with pytest.raises(OSError):
        save_rois(path, draft, overwrite=True)
    assert path.read_bytes() == before and not list(tmp_path.glob(".roi-*.tmp"))
    with pytest.raises(CalibrationError, match="read_only"):
        save_rois("configs/roi.json", draft, overwrite=True)
    with pytest.raises(CalibrationError, match="json"):
        save_rois(tmp_path / "source.mov", draft)


def test_unicode_image_read_and_invalid_inputs(tmp_path):
    image = np.full((60, 120, 3), 80, np.uint8)
    for suffix in (".jpg", ".png"):
        path = tmp_path / f"кадр{suffix}"
        path.write_bytes(cv2.imencode(suffix, image)[1].tobytes())
        loaded, frame = load_calibration_frame(path)
        assert loaded.shape == image.shape and frame is None
        assert np.allclose(image, loaded, atol=2)
        with pytest.raises(CalibrationError, match="requires_mov"):
            load_calibration_frame(path, 1)
    with pytest.raises(CalibrationError, match="missing"):
        load_calibration_frame(tmp_path / "missing.png")
    path.write_bytes(b"bad")
    with pytest.raises(CalibrationError, match="invalid_local_image"):
        load_calibration_frame(path)


def test_pts_selected_mov_vfr_and_nonzero_origin(tmp_path):
    path = make_video(tmp_path / "vfr.mov", times=[1000, 1100, 1400, 1900, 2300])
    before = path.read_bytes()
    image, frame = load_calibration_frame(path, 0.25)
    assert frame.timestamp_sec == pytest.approx(0.4)
    assert frame.source_time_sec == pytest.approx(1.4)
    assert image.shape == (96, 128, 3) and path.read_bytes() == before
    with pytest.raises(CalibrationError, match="beyond"):
        load_calibration_frame(path, 99)
    for bad in [-1, float("nan"), float("inf")]:
        with pytest.raises(CalibrationError, match="invalid_time"):
            load_calibration_frame(path, bad)
    path.write_bytes(b"bad")
    with pytest.raises(VideoError):
        load_calibration_frame(path)


def test_editor_click_drag_undo_scale_and_confirmation(tmp_path):
    image = np.zeros((401, 801, 3), np.uint8)
    session = CalibrationSession("camera-1")
    editor = ROIEditor(image, session, tmp_path / "roi.json", max_side=400)
    for x, y in [(20, 20), (380, 20), (380, 180), (20, 180)]:
        editor.mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        editor.mouse(cv2.EVENT_LBUTTONUP, x, y, 0, None)
    assert len(session.points[session.selected]) == 4
    before = list(session.points[session.selected])
    editor.key(ord("+"))
    assert session.points[session.selected] == before
    assert editor.render().shape[0] == editor.transform.height + 184
    px, py = editor.transform.pixel(before[0])
    editor.mouse(cv2.EVENT_LBUTTONDOWN, px, py, 0, None)
    editor.mouse(cv2.EVENT_MOUSEMOVE, px + 10, py + 10, 0, None)
    editor.mouse(cv2.EVENT_LBUTTONUP, px + 10, py + 10, 0, None)
    assert session.points[session.selected][0] != before[0]
    editor.key(ord("c"))
    assert not editor.pending and not editor.output.exists()
    editor.key(ord("s"))
    assert not load_rois(editor.output).calibrated
    editor.key(ord("p"))
    editor.key(ord("c"))
    assert editor.pending and not load_rois(editor.output).calibrated
    editor.key(ord("n"))
    assert not editor.pending
    editor.key(ord("c"))
    editor.key(ord("y"))
    assert load_rois(editor.output).active("camera-1")
    editor.key(ord("p"))
    editor.key(ord("u"))
    assert len(session.points[session.selected]) == 3 and not session.reviewed
    editor.key(ord("r"))
    assert not session.points[session.selected]


def test_preview_is_read_only_and_toolbar_is_not_a_point(tmp_path):
    session = drawn_session()
    editor = ROIEditor(np.zeros((100, 100, 3), np.uint8), session, tmp_path / "x.json")
    editor.mouse(cv2.EVENT_LBUTTONDOWN, 30, 101, 0, None)
    assert session.points[session.selected] == SQUARE
    editor.read_only = True
    for key in "rcysu":
        editor.key(ord(key))
    editor.mouse(cv2.EVENT_LBUTTONDOWN, 30, 30, 0, None)
    assert session.points[session.selected] == SQUARE and not editor.output.exists()


def test_gui_preflight_never_opens_window_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: pytest.fail("native GUI called in CI"))
    with pytest.raises(CalibrationError, match="GUI unavailable"):
        require_gui()
    monkeypatch.setenv("CI", "false")
    monkeypatch.setattr(cv2, "getBuildInformation", lambda: "  GUI: NONE\n")
    with pytest.raises(CalibrationError, match="opencv-python"):
        require_gui()


def test_cli_preflight_exit_and_no_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CI", "true")
    output = tmp_path / "out.json"
    assert main(["missing.mov", "--camera-profile", "camera-1", "--output", str(output)]) == 2
    assert "GUI unavailable" in capsys.readouterr().err and not output.exists()
    save_rois(output, drawn_session().configuration())
    assert main(["missing.mov", "--camera-profile", "camera-1", "--output", str(output)]) == 2
    assert "overwrite" in capsys.readouterr().err
    assert (
        main(["missing.mov", "--camera-profile", "camera-2", "--load", str(output), "--preview"])
        == 2
    )
    assert "mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("mov", [False, True])
def test_cli_local_source_review_save_preview_with_mock_window(tmp_path, monkeypatch, mov):
    """Exercise the CLI/event loop; only the native window functions are mocked."""
    draft = tmp_path / "draft.json"
    save_rois(draft, drawn_session().configuration())
    if mov:
        source = make_video(tmp_path / "control.mov", times=[1000, 1100, 1400, 1900])
        time_args = ["--time-sec", "0.25"]
    else:
        source = tmp_path / "кадр.jpg"
        source.write_bytes(cv2.imencode(".jpg", np.zeros((96, 128, 3), np.uint8))[1].tobytes())
        time_args = []
    before = source.read_bytes()
    output = tmp_path / "active.json"
    calls = []
    keys = iter(map(ord, "pcyq"))
    monkeypatch.setattr("scripts.calibrate_roi.require_gui", lambda: None)
    monkeypatch.setattr("scripts.roi_editor.require_gui", lambda: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *args: calls.append("open"))
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: calls.append(image.shape))
    monkeypatch.setattr(cv2, "waitKeyEx", lambda delay: next(keys))
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *args: calls.append("close"))
    args = [str(source), "--camera-profile", "camera-1", "--load", str(draft), *time_args]
    assert main([*args, "--output", str(output)]) == 0
    assert load_rois(output).active("camera-1")
    assert source.read_bytes() == before and calls[0] == "open" and calls[-1] == "close"
    active_before = output.read_bytes()
    keys = iter([ord("q")])
    assert (
        main(
            [
                str(source),
                "--camera-profile",
                "camera-1",
                "--load",
                str(output),
                "--preview",
                *time_args,
            ]
        )
        == 0
    )
    assert output.read_bytes() == active_before
    # A successful earlier save must not hide later unsaved edits on exit.
    keys = iter(map(ord, "surq"))
    assert main([*args, "--output", str(output), "--overwrite"]) == 2


def test_native_gui_error_is_sanitized_and_not_a_hang(tmp_path, monkeypatch, capsys):
    source = tmp_path / "control.png"
    source.write_bytes(cv2.imencode(".png", np.zeros((100, 100, 3), np.uint8))[1].tobytes())
    monkeypatch.setattr("scripts.calibrate_roi.require_gui", lambda: None)
    monkeypatch.setattr("scripts.roi_editor.require_gui", lambda: None)

    def fail_window(*args):
        raise cv2.error("private native details")

    monkeypatch.setattr(cv2, "namedWindow", fail_window)
    output = tmp_path / "out.json"
    assert main([str(source), "--camera-profile", "camera-1", "--output", str(output)]) == 2
    error = capsys.readouterr().err
    assert "GUI unavailable" in error and "private" not in error and not output.exists()


def test_multiple_rois_motion_masks_and_outside_exclusion(tmp_path):
    session = CalibrationSession("camera-1")
    session.select("desk")
    session.change([(0, 0), (0.35, 0), (0.35, 1), (0, 1)])
    session.select("cash_area")
    session.change([(0.65, 0), (1, 0), (1, 1), (0.65, 1)])
    session.reviewed = True
    path = tmp_path / "roi.json"
    save_rois(path, session.configuration(confirm=True))
    rois = load_rois(path).active("camera-1")
    assert rois[0].mask(101, 101)[50, 50] == 0
    assert rois[1].mask(101, 101)[50, 85] == 255
    detector = MotionDetector(MotionConfig(blur_size=1), rois)
    black = np.zeros((101, 101, 3), np.uint8)
    detector.update(black, 0)
    middle = black.copy()
    middle[20:60, 45:55] = 255
    assert not detector.update(middle, 1)
    both = middle.copy()
    both[20:60, 5:25] = both[20:60, 75:95] = 255
    assert {hit.roi_id for hit in detector.update(both, 2)} == {"desk", "cash_area"}


def test_calibrated_pipeline_crops_vlm_and_guards_glass(video, tmp_path, sdk, monkeypatch):
    session = CalibrationSession("camera-1")
    session.select("glass_partition")
    session.change([(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)])
    session.reviewed = True
    path = tmp_path / "roi.json"
    save_rois(path, session.configuration(confirm=True))
    config = VisionConfig(roi_path=path, camera_profile="camera-1", output_dir=tmp_path / "out")
    sdk.payload.update(people_visible="yes", physical_presence="yes", reflection_possible="no")
    seen = []
    original_crop = ROI.crop

    def crop(self, image, context=0):
        result = original_crop(self, image, context)
        seen.append((self.roi_id, context, image.shape, result.shape))
        return result

    monkeypatch.setattr(ROI, "crop", crop)
    result = _analyze(
        video, config, time.monotonic() + 30, client=OllamaVisionClient(config.vlm, sdk)
    )
    assert result.status == "complete" and result.events
    assert seen and all(item[0] == "glass_partition" and item[1] == 0.1 for item in seen)
    assert all(item[3][0] < item[2][0] and item[3][1] < item[2][1] for item in seen)
    for event in result.events:
        assert event.roi_id == "glass_partition"
        assert event.observations["physical_presence"] == "unknown"
        assert event.observations["reflection_possible"] == "yes"
        assert "rule_id" not in json.loads(event.model_dump_json())
    config.camera_profile = "other-camera"
    seen.clear()
    result = _analyze(
        video, config, time.monotonic() + 30, client=OllamaVisionClient(config.vlm, sdk)
    )
    assert result.events and all(event.roi_id is None for event in result.events)
    assert not seen and any(d.code == "roi_classification_unavailable" for d in result.diagnostics)
