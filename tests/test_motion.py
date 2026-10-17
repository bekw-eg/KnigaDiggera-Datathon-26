import numpy as np
import pytest

from app.vision.motion import MotionConfig, MotionDetector, MotionInterval, merge_intervals
from app.vision.roi import ROI


def test_no_motion_and_moving_object():
    detector = MotionDetector(MotionConfig())
    black = np.zeros((100, 100, 3), np.uint8)
    assert detector.update(black, 0) == []
    assert detector.update(black, 0.5) == []
    changed = black.copy()
    changed[20:60, 20:60] = 255
    hits = detector.update(changed, 1)
    assert len(hits) == 1 and hits[0].motion_score > 0.1
    assert hits[0].start_sec == 0.5
    assert "violation" not in hits[0].model_dump()


def test_roi_excludes_unrelated_motion():
    roi = ROI(roi_id="left", polygon=[(0, 0), (0.4, 0), (0.4, 1), (0, 1)])
    detector = MotionDetector(MotionConfig(), [roi])
    black = np.zeros((100, 100, 3), np.uint8)
    detector.update(black, 0)
    changed = black.copy()
    changed[20:60, 70:95] = 255
    assert detector.update(changed, 1) == []


def test_merge_gap_preroll_postroll_and_roi_separation():
    items = [
        MotionInterval(start_sec=a, end_sec=b, roi_id=roi, motion_score=0.1)
        for a, b, roi in [(1, 2, "desk"), (2.3, 3, "desk"), (8, 9, "desk"), (2, 3, "glass")]
    ]
    output = merge_intervals(
        items, MotionConfig(merge_gap_sec=0.2, pre_roll_sec=0.1, post_roll_sec=0.1), end_sec=9
    )
    assert len(output) == 3
    assert output[0].start_sec == pytest.approx(0.9)
    assert output[0].end_sec == pytest.approx(3.1)
    assert output[-1].end_sec == 9
