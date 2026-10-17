"""Diagnostic envelope around the unchanged shared CandidateEvent contract."""

from typing import Literal

from pydantic import Field

from app.schemas import CandidateEvent, ContractModel, Seconds
from app.vision.evidence import EventReplay
from app.vision.video import VideoMetadata


class Diagnostic(ContractModel):
    code: str
    severity: Literal["warning", "error"] = "warning"
    event_id: str | None = None


class EvidenceFrame(ContractModel):
    path: str
    timestamp_sec: Seconds
    source_pts: int
    time_base: str
    source_time_sec: float


class VisionMetrics(ContractModel):
    decoded_frames: int = 0
    sampled_frames: int = 0
    first_pass_decoded_frames: int = 0
    second_pass_decoded_frames: int = 0
    second_pass_frames: int = 0
    candidates_before_dedup: int = 0
    candidates_after_dedup: int = 0
    refined_candidates: int = 0
    candidates_total: int = 0
    vlm_calls: int = 0
    vlm_successes: int = 0
    decode_sec: Seconds = 0
    cv_sec: Seconds = 0
    evidence_io_sec: Seconds = 0
    vlm_sec: Seconds = 0
    total_sec: Seconds = 0


class Coverage(ContractModel):
    decoded_until_sec: Seconds | None = None
    decoded_complete: bool = False
    decoded_fraction: float | None = Field(default=None, ge=0, le=1)
    sample_fps: float
    vlm_intervals: list[tuple[Seconds, Seconds]] = Field(default_factory=list)
    refine_fps: float | None = None
    refined_intervals: list[tuple[Seconds, Seconds]] = Field(default_factory=list)


class VisionAnalysisResult(ContractModel):
    status: Literal["complete", "partial", "failed", "timed_out"] = "partial"
    events: list[CandidateEvent] = Field(default_factory=list)
    metadata: VideoMetadata | None = None
    evidence_frames: list[EvidenceFrame] = Field(default_factory=list)
    event_replay: list[EventReplay] = Field(default_factory=list)
    metrics: VisionMetrics = Field(default_factory=VisionMetrics)
    coverage: Coverage
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    output_dir: str | None = None
