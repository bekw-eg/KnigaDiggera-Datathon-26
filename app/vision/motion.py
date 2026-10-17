"""CPU frame differencing; scores measure changed pixels, never violations."""

import cv2
import numpy as np
from pydantic import Field, field_validator, model_validator

from app.schemas import ContractModel, Seconds
from app.vision.roi import ROI
from app.vision.video import Image


class MotionConfig(ContractModel):
    pixel_threshold: int = Field(default=25, ge=1, le=255)
    min_contour_area: float = Field(default=24, ge=0)
    min_area_fraction: float = Field(default=0.003, gt=0, le=1)
    blur_size: int = Field(default=5, ge=1, le=31)
    merge_gap_sec: Seconds = 1.0
    pre_roll_sec: Seconds = 1.0
    post_roll_sec: Seconds = 1.0

    @field_validator("blur_size")
    @classmethod
    def odd_blur(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("blur_size must be odd")
        return value


class MotionInterval(ContractModel):
    start_sec: Seconds
    end_sec: Seconds
    roi_id: str | None = None
    motion_score: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> "MotionInterval":
        if self.start_sec > self.end_sec:
            raise ValueError("motion interval start exceeds end")
        return self


class MotionDetector:
    """Streaming consecutive sampled-frame comparison using optional calibrated ROIs."""

    def __init__(self, config: MotionConfig, rois: list[ROI] | None = None):
        if config.blur_size % 2 != 1:
            raise ValueError("blur_size must be odd")
        self.config = config
        self.rois = rois or []
        self.previous: np.ndarray | None = None
        self.previous_sec: float | None = None

    def update(self, image: Image, timestamp_sec: float) -> list[MotionInterval]:
        """Emit motion between previous/current sample; caller adds context/merging."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.config.blur_size, self.config.blur_size), 0)
        previous, previous_sec = self.previous, self.previous_sec
        self.previous, self.previous_sec = gray, timestamp_sec
        if previous is None:
            return []
        if previous.shape != gray.shape or timestamp_sec <= previous_sec:
            raise ValueError("frame shape/time changed")
        delta = cv2.absdiff(gray, previous)
        _, binary = cv2.threshold(delta, self.config.pixel_threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        height, width = gray.shape
        regions = [(roi.roi_id, roi.mask(width, height)) for roi in self.rois]
        if not regions:
            regions = [(None, np.full(gray.shape, 255, dtype=np.uint8))]
        hits = []
        for roi_id, mask in regions:
            selected = cv2.bitwise_and(binary, mask)
            contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = np.zeros_like(selected)
            for contour in contours:
                if cv2.contourArea(contour) >= self.config.min_contour_area:
                    cv2.drawContours(filtered, [contour], -1, 255, cv2.FILLED)
            score = np.count_nonzero(cv2.bitwise_and(filtered, mask)) / max(
                1, np.count_nonzero(mask)
            )
            if score >= self.config.min_area_fraction:
                hits.append(
                    MotionInterval(
                        start_sec=previous_sec,
                        end_sec=timestamp_sec,
                        roi_id=roi_id,
                        motion_score=float(score),
                    )
                )
        return hits


def merge_intervals(
    intervals: list[MotionInterval],
    config: MotionConfig,
    end_sec: float,
) -> list[MotionInterval]:
    """Pad with pre/post-roll and merge nearby intervals only within the same ROI."""
    merged: list[MotionInterval] = []
    keys = sorted({item.roi_id for item in intervals}, key=lambda key: key or "")
    for roi_id in keys:
        group: list[MotionInterval] = []
        for interval in sorted(
            (i for i in intervals if i.roi_id == roi_id), key=lambda i: i.start_sec
        ):
            item = interval.model_copy(
                update={
                    "start_sec": max(0.0, interval.start_sec - config.pre_roll_sec),
                    "end_sec": min(end_sec, interval.end_sec + config.post_roll_sec),
                }
            )
            if group and item.start_sec <= group[-1].end_sec + config.merge_gap_sec:
                group[-1].end_sec = max(group[-1].end_sec, item.end_sec)
                group[-1].motion_score = max(group[-1].motion_score, item.motion_score)
            else:
                group.append(item)
        merged.extend(group)
    return sorted(merged, key=lambda i: (i.start_sec, i.roi_id or ""))
