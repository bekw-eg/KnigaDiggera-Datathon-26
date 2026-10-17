import numpy as np
import pytest
from pydantic import ValidationError

from app.vision.roi import ROI, ROIConfig, load_rois


def test_normalized_pixels_mask_crop_and_context():
    roi = ROI(roi_id="desk", polygon=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    assert roi.pixels(101, 101).tolist() == [[25, 25], [75, 25], [75, 75], [25, 75]]
    assert roi.mask(101, 101)[50, 50] == 255
    assert roi.mask(101, 101)[0, 0] == 0
    image = np.ones((101, 101, 3), np.uint8)
    assert roi.crop(image).shape == (51, 51, 3)
    assert roi.crop(image, context=0.5).shape == image.shape
    assert roi.box_overlap((30, 30, 60, 60), 101, 101) == 1
    assert roi.box_overlap((-10, -10, 0, 0), 101, 101) == 0
    assert roi.box_overlap((90, 90, 100, 100), 101, 101) == 0


@pytest.mark.parametrize(
    "points",
    [
        [(0, 0), (2, 0), (1, 1)],
        [(0, 0), (0, 0), (1, 1)],
        [(0, 0), (0.5, 0.5), (1, 1)],
        [(0, 0), (1, 1), (0, 1), (1, 0)],
        [(0, 0), (float("nan"), 0), (1, 1)],
    ],
)
def test_invalid_polygon(points):
    with pytest.raises(ValidationError):
        ROI(roi_id="bad", polygon=points)


def test_calibration_gate():
    demo = load_rois("configs/roi.json")
    assert not demo.active("DEMO_UNCALIBRATED")
    calibrated = ROIConfig(
        camera_profile="camera-1", calibrated=True, demonstration=False, rois=demo.rois
    )
    assert not calibrated.active("camera-2")
    assert calibrated.active("camera-1")
    with pytest.raises(ValidationError):
        ROIConfig(camera_profile="x", calibrated=True, demonstration=True, rois=[])
