"""Local calibration primitives: display coordinates, PTS selection and atomic JSON.

No GUI or inference is started on import. Geometry uses ROI/ROIConfig, the same
models consumed by the pipeline. A drawn polygon alone never activates calibration.
"""

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.vision.roi import ROI, Point, ROIConfig
from app.vision.video import Image, VideoFrame, VideoReader

ZONE_IDS = (
    "main_work_area",
    "desk",
    "glass_partition",
    "transfer_zone",
    "cash_area",
    "safe_area",
)


class CalibrationError(RuntimeError):
    """Explicit local calibration failure; messages contain no media/model data."""


@dataclass(frozen=True)
class DisplayTransform:
    """Map the displayed image's inclusive pixel grid to normalized ROI vertices.

    The editor controls scaling with +/- in an autosized window. Mouse coordinates
    therefore always refer to this exact raster, without window borders/letterbox.
    Width/height rounding is handled independently, including non-square images.
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("display dimensions must be at least 2")

    @classmethod
    def fit(cls, width: int, height: int, max_side: int) -> "DisplayTransform":
        """Fit a source frame without upscaling; keep both axes invertible."""
        if width < 2 or height < 2 or max_side < 2:
            raise ValueError("invalid source/display dimensions")
        scale = min(1.0, max_side / max(width, height))
        return cls(max(2, round(width * scale)), max(2, round(height * scale)))

    def normalize(self, x: float, y: float) -> Point:
        """Normalize a mouse point; never silently clamp off-image coordinates."""
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("nonfinite display point")
        if not 0 <= x <= self.width - 1 or not 0 <= y <= self.height - 1:
            raise ValueError("point outside displayed image")
        return x / (self.width - 1), y / (self.height - 1)

    def pixel(self, point: Point) -> tuple[int, int]:
        """Map a normalized vertex to the displayed image (same rule as ROI)."""
        x, y = point
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError("invalid normalized point")
        return round(x * (self.width - 1)), round(y * (self.height - 1))


def load_calibration_frame(path: Path, time_sec: float = 0) -> tuple[Image, VideoFrame | None]:
    """Read JPEG/PNG, or the first MOV frame at/after a relative PTS time in seconds.

    VideoReader validates PTS of skipped frames too, including VFR/nonzero origins.
    Decode is sequential, so late timecodes may take time. No frame is exported.
    Video orientation matches the pipeline's decoded raster (no auto-rotation).
    """
    if not math.isfinite(time_sec) or time_sec < 0:
        raise CalibrationError("invalid_time_sec")
    path = Path(path)
    if not path.is_file():
        raise CalibrationError("input_file_missing")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        if time_sec != 0:
            raise CalibrationError("time_sec_requires_mov")
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise CalibrationError("invalid_local_image")
        if min(image.shape[:2]) < 2 or image.shape[0] * image.shape[1] > 40_000_000:
            raise CalibrationError("invalid_or_excessive_resolution")
        return image, None
    if path.suffix.lower() != ".mov":
        raise CalibrationError("input_must_be_jpeg_png_or_mov")
    with VideoReader(path) as reader:
        frame = next(
            reader.sample(
                start_sec=time_sec, max_side=max(reader.metadata.width, reader.metadata.height)
            ),
            None,
        )
    if frame is None:
        raise CalibrationError("time_sec_beyond_last_frame")
    return frame.image, frame


def save_rois(path: Path, config: ROIConfig, *, overwrite: bool = False) -> None:
    """Validate and publish UTF-8 JSON atomically beside its destination.

    Existing files require explicit overwrite. Failed writes leave the old file
    intact. The repository's demonstration config is protected even with overwrite.
    No image, source filename or evidence is persisted in this document.
    """
    config = ROIConfig.model_validate(config.model_dump(mode="json"))
    path = Path(path).resolve()
    demo_path = Path(__file__).resolve().parents[2] / "configs" / "roi.json"
    if path == demo_path:
        raise CalibrationError("repository_demo_config_is_read_only")
    if path.suffix.lower() != ".json":
        raise CalibrationError("output_must_be_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".roi-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(config.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic create-if-absent, including a race with another local writer.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CalibrationSession:
    """Editable polygons; every save revalidates and defaults to an inactive draft."""

    def __init__(
        self,
        camera_profile: str,
        config: ROIConfig | None = None,
        *,
        demonstration: bool = False,
        safe_visible: bool = False,
    ) -> None:
        ROIConfig(camera_profile=camera_profile, rois=[])
        if config is not None and config.camera_profile != camera_profile:
            raise CalibrationError("camera_profile_mismatch")
        if config is not None and any(roi.roi_id not in ZONE_IDS for roi in config.rois):
            raise CalibrationError("unsupported_calibration_zone")
        self.camera_profile = camera_profile
        self.demonstration = demonstration or (config is not None and config.demonstration)
        self.safe_visible = safe_visible
        self.selected = ZONE_IDS[0]
        self.points: dict[str, list[Point]] = (
            {roi.roi_id: list(roi.polygon) for roi in config.rois} if config else {}
        )
        self.reviewed = False
        self.revision = 0

    def select(self, zone_id: str) -> None:
        """Select an allowed zone; safe geometry requires an explicit visibility claim."""
        if zone_id not in ZONE_IDS:
            raise CalibrationError("unsupported_calibration_zone")
        if zone_id == "safe_area" and not self.safe_visible:
            raise CalibrationError("safe_area_requires_safe_visible")
        self.selected = zone_id

    def change(self, points: list[Point]) -> None:
        """Replace the selected draft, invalidating any earlier visual review."""
        if self.selected == "safe_area" and not self.safe_visible:
            raise CalibrationError("safe_area_requires_safe_visible")
        if len(points) > 100:
            raise CalibrationError("polygon_vertex_limit")
        self.points[self.selected] = list(points)
        self.reviewed = False
        self.revision += 1

    def configuration(self, *, confirm: bool = False) -> ROIConfig:
        """Build a draft, or activate only after explicit review and confirmation.

        The GUI alone calls confirm=True after showing all ROIs, the camera profile
        and a second Y keypress. Demonstration configs can never be activated.
        """
        rois = [ROI(roi_id=key, polygon=points) for key, points in self.points.items() if points]
        if not rois:
            raise CalibrationError("draw_at_least_one_zone")
        if any(roi.roi_id == "safe_area" for roi in rois) and not self.safe_visible:
            raise CalibrationError("safe_area_requires_safe_visible")
        if confirm and (not self.reviewed or self.demonstration):
            raise CalibrationError("visual_review_required_or_demonstration")
        return ROIConfig(
            camera_profile=self.camera_profile,
            calibrated=confirm,
            demonstration=self.demonstration,
            rois=rois,
        )
