"""Typed replay data and conservative selection of local before/during/after frames."""

import hashlib
from typing import Literal

import cv2
from pydantic import Field, model_validator

from app.schemas import ContractModel, Observation, Seconds
from app.vision.roi import Point
from app.vision.temporal import TemporalProposal
from app.vision.video import Image


class ReplayFrame(ContractModel):
    path: str
    timestamp_sec: Seconds
    source_pts: int
    time_base: str
    source_time_sec: float
    role: Literal["before", "during", "after"]
    fingerprint: str
    vlm_crop_roi_id: str | None = None
    vlm_crop_polygon: list[Point] | None = None
    vlm_crop_context: float = Field(default=0.1, ge=0, le=1)
    vlm_max_side: int = Field(default=1024, ge=64, le=2048)


class EventReplay(ContractModel):
    """Local UI sidecar. Observations are sanitized tri-state fields, never free text."""

    event_id: str
    start_sec: Seconds
    end_sec: Seconds
    roi_id: str | None = None
    roi_ids: list[str] = Field(default_factory=list)
    peak_sec: Seconds
    segmentation_insufficient: bool = False
    temporal_complete: bool = False
    boundary_activity: bool = False
    status: Literal["pending", "verified", "incomplete", "skipped_budget", "failed"] = "pending"
    frames: list[ReplayFrame] = Field(default_factory=list)
    observations: dict[str, Observation] = Field(default_factory=dict)
    diagnostic_codes: list[str] = Field(default_factory=list)
    vlm_frame_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_evidence(self) -> "EventReplay":
        times = [f.timestamp_sec for f in self.frames]
        if not self.start_sec <= self.peak_sec <= self.end_sec or times != sorted(set(times)):
            raise ValueError("invalid replay chronology")
        if len({f.role for f in self.frames}) != len(self.frames):
            raise ValueError("duplicate replay roles")
        if any(path not in {f.path for f in self.frames} for path in self.vlm_frame_paths):
            raise ValueError("unknown VLM evidence path")
        if any(t < self.start_sec or t > self.end_sec for t in times):
            raise ValueError("frame outside replay window")
        if self.temporal_complete and (
            self.segmentation_insufficient
            or [f.role for f in self.frames] != ["before", "during", "after"]
            or len({f.fingerprint for f in self.frames}) != 3
        ):
            raise ValueError("incomplete temporal evidence")
        return self


def frame_fingerprint(image: Image) -> str:
    """Coarse grayscale signature rejects repeated decoded/static frames, not identity."""
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY)
    return hashlib.sha256((gray // 8).tobytes()).hexdigest()


def select_temporal_frames(
    frames: list[ReplayFrame], proposal: TemporalProposal
) -> tuple[list[ReplayFrame], bool]:
    """Select actual context around measured activity; never relabel a missing role.

    Nearest peak is used for during; before/after must bracket activity and have
    different coarse signatures. Missing context keeps only available roles.
    """
    within = [f for f in frames if proposal.start_sec <= f.timestamp_sec <= proposal.end_sec]
    during = sorted(
        (
            f
            for f in within
            if proposal.activity_start_sec < f.timestamp_sec <= proposal.activity_end_sec
        ),
        key=lambda f: abs(f.timestamp_sec - proposal.peak_sec),
    )
    if not during:
        return [], False
    context = (
        (
            "before",
            sorted(
                (
                    f
                    for f in within
                    if proposal.before_observed
                    and f.timestamp_sec <= proposal.activity_start_sec
                    and (
                        proposal.before_sec is None
                        or abs(f.timestamp_sec - proposal.before_sec) < 1e-8
                    )
                ),
                key=lambda f: -f.timestamp_sec,
            ),
        ),
        (
            "after",
            sorted(
                (
                    f
                    for f in within
                    if proposal.after_observed
                    and f.timestamp_sec > proposal.activity_end_sec
                    and (
                        proposal.after_sec is None
                        or abs(f.timestamp_sec - proposal.after_sec) < 1e-8
                    )
                ),
                key=lambda f: f.timestamp_sec,
            ),
        ),
    )
    best: list[ReplayFrame] = []
    for peak_frame in during:
        selected = [peak_frame.model_copy(update={"role": "during"})]
        for role, candidates in context:
            frame = next(
                (
                    f
                    for f in candidates
                    if all(f.fingerprint != other.fingerprint for other in selected)
                ),
                None,
            )
            if frame is not None:
                selected.append(frame.model_copy(update={"role": role}))
        selected.sort(key=lambda f: f.timestamp_sec)
        if len(selected) > len(best):
            best = selected
        if len(best) == 3:
            break
    return best, not proposal.segmentation_insufficient and len(best) == 3
