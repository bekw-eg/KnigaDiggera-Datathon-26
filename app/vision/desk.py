"""Calibrated desk motion, intersecting zones and optional boundary geometry."""

import cv2
import numpy as np

from app.vision.motion import MotionConfig, MotionInterval
from app.vision.roi import ROI
from app.vision.temporal import TransferBoundary
from app.vision.video import Image


class DeskMotionAnalyzer:
    """Score every sample (including quiet ones), clipping zones to the desk mask.

    With no calibrated desk, existing active ROIs/full frame are the fallback. The
    caller must report that fallback. Boundary activity means changed pixels on both
    sides near a finite segment, never a hand/object crossing or cash transfer.
    """

    def __init__(
        self,
        config: MotionConfig,
        rois: list[ROI],
        desk_roi_id: str = "desk",
        boundary: TransferBoundary | None = None,
    ) -> None:
        self.config = config
        self.rois = rois
        self.desk = next((r for r in rois if r.roi_id == desk_roi_id), None)
        self.boundary = boundary if self.desk is not None else None
        self.previous: np.ndarray | None = None
        self.previous_sec: float | None = None
        self.boundary_activity = False
        self._masks: list[tuple[str | None, np.ndarray]] | None = None

    def reset(self) -> None:
        """Do not compare frames across unobserved gaps in second-pass windows."""
        self.previous = None
        self.previous_sec = None

    def update(self, image: Image, timestamp_sec: float) -> list[MotionInterval]:
        """Measure changed-pixel fractions using real consecutive PTS pairs."""
        gray = cv2.GaussianBlur(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            (self.config.blur_size, self.config.blur_size),
            0,
        )
        previous, previous_sec = self.previous, self.previous_sec
        self.previous, self.previous_sec = gray, timestamp_sec
        self.boundary_activity = False
        if previous is None:
            return []
        if gray.shape != previous.shape or timestamp_sec <= previous_sec:
            raise ValueError("frame_shape_or_pts_changed")
        binary = cv2.threshold(
            cv2.absdiff(gray, previous), self.config.pixel_threshold, 255, cv2.THRESH_BINARY
        )[1]
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        height, width = gray.shape
        if self._masks is None:
            desk_mask = self.desk.mask(width, height) if self.desk else None
            self._masks = []
            for roi in self.rois:
                mask = roi.mask(width, height)
                if desk_mask is not None:
                    mask &= desk_mask
                if np.any(mask):
                    self._masks.append((roi.roi_id, mask))
            if not self._masks:
                self._masks = [(None, np.full(gray.shape, 255, np.uint8))]
        scores = []
        for roi_id, mask in self._masks:
            selected = binary & mask
            contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = np.zeros_like(selected)
            for contour in contours:
                if cv2.contourArea(contour) >= self.config.min_contour_area:
                    cv2.drawContours(filtered, [contour], -1, 255, cv2.FILLED)
            scores.append(
                MotionInterval(
                    start_sec=previous_sec,
                    end_sec=timestamp_sec,
                    roi_id=roi_id,
                    motion_score=float(np.count_nonzero(filtered & mask) / np.count_nonzero(mask)),
                )
            )
        if self.boundary and self.boundary.enabled:
            a, b = np.asarray(self.boundary.points)
            y, x = np.nonzero(binary & self.desk.mask(width, height))
            points = np.column_stack((x / max(1, width - 1), y / max(1, height - 1)))
            vector = b - a
            length = np.linalg.norm(vector)
            relative = points - a
            projection = relative @ vector / (length * length)
            signed = (vector[0] * relative[:, 1] - vector[1] * relative[:, 0]) / length
            near = (
                (projection >= 0) & (projection <= 1) & (np.abs(signed) <= self.boundary.band_width)
            )
            self.boundary_activity = bool(np.any(signed[near] < 0) and np.any(signed[near] > 0))
        return scores
