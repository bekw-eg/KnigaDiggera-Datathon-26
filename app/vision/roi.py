"""Camera-specific normalized polygon geometry; demo calibration is never active."""

from pathlib import Path
from typing import Annotated, Self

import cv2
import numpy as np
from pydantic import Field, model_validator

from app.schemas import ContractModel
from app.vision.video import Image

Coordinate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Point = tuple[Coordinate, Coordinate]
Box = tuple[float, float, float, float]


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a: Point, b: Point, c: Point, d: Point) -> bool:
    if max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]):
        return False
    if max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1]):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


class ROI(ContractModel):
    roi_id: str = Field(min_length=1)
    polygon: list[Point] = Field(min_length=3, max_length=100)

    @model_validator(mode="after")
    def valid_polygon(self) -> Self:
        points = self.polygon
        count = len(points)
        if len(set(points)) != count:
            raise ValueError("polygon has duplicate vertices")
        area = (
            abs(
                sum(
                    points[i][0] * points[(i + 1) % count][1]
                    - points[(i + 1) % count][0] * points[i][1]
                    for i in range(count)
                )
            )
            / 2
        )
        if area < 1e-8:
            raise ValueError("polygon has zero area")
        for i in range(count):
            for j in range(i + 1, count):
                if j == i + 1 or (i == 0 and j == count - 1):
                    continue
                if _intersects(
                    points[i], points[(i + 1) % count], points[j], points[(j + 1) % count]
                ):
                    raise ValueError("polygon self-intersects")
        return self

    def pixels(self, width: int, height: int) -> np.ndarray:
        """Map inclusive normalized vertices onto valid pixel indices."""
        if width < 1 or height < 1:
            raise ValueError("invalid image dimensions")
        return np.rint(np.array(self.polygon) * [width - 1, height - 1]).astype(np.int32)

    def mask(self, width: int, height: int) -> np.ndarray:
        """Return a uint8 polygon mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.pixels(width, height)], 255)
        return mask

    def box_overlap(self, box: Box, width: int, height: int) -> float:
        """Fraction of the clipped bounding box lying inside this ROI (not probability)."""
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        return float(np.count_nonzero(self.mask(width, height)[y1:y2, x1:x2])) / (
            (x2 - x1) * (y2 - y1)
        )

    def crop(self, image: Image, context: float = 0.0) -> Image:
        """Masked ROI crop, or unmasked surrounding context when context > 0."""
        if not 0 <= context <= 1:
            raise ValueError("context must be 0..1")
        height, width = image.shape[:2]
        points = self.pixels(width, height)
        x, y, w, h = cv2.boundingRect(points)
        dx, dy = round(w * context), round(h * context)
        source = image if context else cv2.bitwise_and(image, image, mask=self.mask(width, height))
        return source[
            max(0, y - dy) : min(height, y + h + dy), max(0, x - dx) : min(width, x + w + dx)
        ].copy()


class ROIConfig(ContractModel):
    camera_profile: str
    calibrated: bool = False
    demonstration: bool = True
    rois: list[ROI]

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [roi.roi_id for roi in self.rois]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ROI ids")
        if self.calibrated and self.demonstration:
            raise ValueError("demonstration coordinates cannot be calibrated")
        return self

    def active(self, camera_profile: str | None) -> list[ROI]:
        """Fail closed for uncalibrated, demonstration or mismatched camera profiles."""
        if not self.calibrated or self.demonstration or camera_profile != self.camera_profile:
            return []
        return self.rois


def load_rois(path: Path) -> ROIConfig:
    """Load and validate a local ROI document; errors are not silently ignored."""
    return ROIConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
