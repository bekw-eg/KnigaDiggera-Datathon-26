"""Pure PTS-based activity proposals. Windows describe observations, never actions."""

from pydantic import Field, model_validator

from app.schemas import ContractModel, Seconds
from app.vision.motion import MotionInterval
from app.vision.roi import ROI, CameraProfile, Point


class TransferBoundary(ContractModel):
    """Manually calibrated finite segment; changed pixels are not tracked objects."""

    enabled: bool = False
    camera_profile: CameraProfile
    points: tuple[Point, Point]
    band_width: float = Field(default=0.04, gt=0, le=0.25)

    @model_validator(mode="after")
    def distinct_points(self) -> "TransferBoundary":
        if sum((a - b) ** 2 for a, b in zip(self.points[0], self.points[1], strict=True)) < 1e-12:
            raise ValueError("boundary endpoints must differ")
        return self


class TemporalConfig(ContractModel):
    enabled: bool = False
    refine_fps: float = Field(default=8, gt=0, le=30)
    high_threshold: float = Field(default=0.012, gt=0, le=1)
    low_threshold: float = Field(default=0.003, gt=0, le=1)
    refine_high_threshold: float | None = Field(default=None, gt=0, le=1)
    refine_low_threshold: float | None = Field(default=None, gt=0, le=1)
    pause_sec: float = Field(default=0.5, gt=0, le=10)
    merge_gap_sec: Seconds = 0.15
    max_window_sec: float = Field(default=4, gt=0, le=20)
    pre_roll_sec: Seconds = 0.75
    post_roll_sec: Seconds = 0.75
    max_proposals: int = Field(default=256, ge=1, le=2000)
    max_refined_frames: int = Field(default=768, ge=3, le=10000)
    desk_roi_id: str = Field(default="desk", min_length=1)
    roi_priorities: dict[str, int] = Field(default_factory=dict)
    dedup_overlap: float = Field(default=0.5, gt=0, le=1)
    boundary: TransferBoundary | None = None

    @model_validator(mode="after")
    def valid_thresholds(self) -> "TemporalConfig":
        if self.low_threshold > self.high_threshold:
            raise ValueError("low threshold exceeds high threshold")
        if (
            self.refine_low_threshold
            if self.refine_low_threshold is not None
            else self.low_threshold
        ) > (
            self.refine_high_threshold
            if self.refine_high_threshold is not None
            else self.high_threshold
        ):
            raise ValueError("refinement low threshold exceeds refinement high threshold")
        if self.pre_roll_sec + self.post_roll_sec >= self.max_window_sec:
            raise ValueError("rolls must leave room for activity")
        if self.merge_gap_sec >= self.pause_sec:
            raise ValueError("merge gap must be shorter than a separating pause")
        return self

    def refinement_config(self) -> "TemporalConfig":
        """Resolve optional second-pass thresholds; never auto-scale scores by FPS."""
        return self.model_copy(
            update={
                "high_threshold": (
                    self.refine_high_threshold
                    if self.refine_high_threshold is not None
                    else self.high_threshold
                ),
                "low_threshold": (
                    self.refine_low_threshold
                    if self.refine_low_threshold is not None
                    else self.low_threshold
                ),
            }
        )


class TemporalProposal(ContractModel):
    start_sec: Seconds
    end_sec: Seconds
    activity_start_sec: Seconds
    activity_end_sec: Seconds
    peak_sec: Seconds
    local_peaks_sec: list[Seconds] = Field(default_factory=list)
    motion_score: float = Field(ge=0, le=1)
    roi_ids: list[str] = Field(default_factory=list)
    roi_id: str | None = None
    segmentation_insufficient: bool = False
    before_observed: bool = True
    after_observed: bool = True
    before_sec: Seconds | None = None
    after_sec: Seconds | None = None

    @model_validator(mode="after")
    def ordered(self) -> "TemporalProposal":
        if not (
            self.start_sec
            <= self.activity_start_sec
            <= self.peak_sec
            <= self.activity_end_sec
            <= self.end_sec
        ):
            raise ValueError("invalid proposal times")
        if any(
            t < self.activity_start_sec or t > self.activity_end_sec for t in self.local_peaks_sec
        ):
            raise ValueError("peak outside activity")
        if (
            self.before_sec is not None
            and not self.start_sec <= self.before_sec <= self.activity_start_sec
        ):
            raise ValueError("before outside context")
        if (
            self.after_sec is not None
            and not self.activity_end_sec < self.after_sec <= self.end_sec
        ):
            raise ValueError("after outside context")
        return self


class ProposalBatch(ContractModel):
    proposals: list[TemporalProposal] = Field(default_factory=list)
    total: int = 0
    truncated: bool = False
    continuous_regions: int = 0


def local_activity_peaks(samples: list[MotionInterval]) -> list[MotionInterval]:
    """Measured local maxima, taking the first sample of a flat maximum plateau."""
    return [
        sample
        for i, sample in enumerate(samples)
        if (i == 0 or sample.motion_score > samples[i - 1].motion_score)
        and (i == len(samples) - 1 or sample.motion_score >= samples[i + 1].motion_score)
    ]


def propose_events(
    samples: list[MotionInterval], config: TemporalConfig, end_sec: float
) -> ProposalBatch:
    """Hysteresis episodes separated by sustained low scores; preserve measured times.

    Long episodes yield bounded windows over actual active samples, explicitly marked
    undersegmented. Their count is not a count of physical actions. All zero/low samples
    must be supplied so a pause is distinguishable from continuous motion.
    """
    batch = ProposalBatch()
    core_limit = config.max_window_sec - config.pre_roll_sec - config.post_roll_sec
    for roi_id in sorted({s.roi_id for s in samples}, key=lambda key: key or ""):
        series = [s for s in samples if s.roi_id == roi_id]
        for previous, current in zip(series, series[1:], strict=False):
            if current.end_sec <= previous.end_sec or current.start_sec < previous.end_sec - 1e-8:
                raise ValueError("nonmonotonic_motion_pts")
        runs: list[list[MotionInterval]] = []
        active: list[MotionInterval] = []
        quiet_start: float | None = None
        for sample in series:
            if active and (
                sample.start_sec - active[-1].end_sec >= config.pause_sec
                or (
                    quiet_start is not None
                    and (
                        sample.start_sec - quiet_start >= config.pause_sec
                        or (
                            sample.end_sec - quiet_start >= config.pause_sec
                            and sample.motion_score < config.low_threshold
                        )
                    )
                )
            ):
                runs.append(active)
                active, quiet_start = [], None
            if sample.motion_score >= config.low_threshold:
                if not active or sample.start_sec - active[-1].end_sec <= config.pause_sec:
                    active.append(sample)
                quiet_start = None
            elif active and quiet_start is None:
                quiet_start = sample.start_sec
        if active:
            runs.append(active)
        # Close fragments can belong to one episode, but established pauses stay separate.
        merged: list[list[MotionInterval]] = []
        for run in runs:
            if merged and run[0].start_sec - merged[-1][-1].end_sec <= config.merge_gap_sec:
                merged[-1].extend(run)
            else:
                merged.append(run)
        for run in merged:
            if max(s.motion_score for s in run) < config.high_threshold:
                continue
            long = run[-1].end_sec - run[0].start_sec > core_limit
            batch.continuous_regions += int(long)
            offset = 0
            while offset < len(run):
                first = run[offset]
                stop = offset + 1
                while stop < len(run) and run[stop].end_sec - first.start_sec <= core_limit:
                    stop += 1
                chunk = run[offset:stop]
                # An unusually sparse pair can exceed the cap: retain its measured endpoint,
                # mark insufficient; do not pretend a missing interior was observed.
                peaks = local_activity_peaks(chunk)
                peak = max(peaks, key=lambda s: s.motion_score)
                activity_start = (
                    first.start_sec
                    if chunk[-1].end_sec - first.start_sec <= core_limit
                    else peak.end_sec
                )
                peak_sec = peak.end_sec
                start = max(0.0, activity_start - config.pre_roll_sec)
                end = min(end_sec, chunk[-1].end_sec + config.post_roll_sec)
                quiet_before = [
                    s.end_sec
                    for s in series
                    if s.motion_score < config.low_threshold
                    and start <= s.end_sec <= first.start_sec
                ]
                quiet_after = [
                    s.end_sec
                    for s in series
                    if s.motion_score < config.low_threshold
                    and chunk[-1].end_sec <= s.start_sec
                    and s.end_sec <= end
                ]
                proposal = TemporalProposal(
                    start_sec=start,
                    end_sec=end,
                    activity_start_sec=activity_start,
                    activity_end_sec=chunk[-1].end_sec,
                    peak_sec=peak_sec,
                    local_peaks_sec=[s.end_sec for s in peaks if s.end_sec >= activity_start],
                    motion_score=peak.motion_score,
                    roi_id=roi_id,
                    roi_ids=[roi_id] if roi_id else [],
                    segmentation_insufficient=long or first.start_sec < activity_start,
                    before_observed=bool(quiet_before),
                    after_observed=bool(quiet_after),
                    before_sec=max(quiet_before) if quiet_before else None,
                    after_sec=min(quiet_after) if quiet_after else None,
                )
                batch.total += 1
                if len(batch.proposals) < config.max_proposals:
                    batch.proposals.append(proposal)
                else:
                    batch.truncated = True
                offset = stop
    return batch


def _spatial_overlap(a: TemporalProposal, b: TemporalProposal, rois: dict[str, ROI]) -> bool:
    if a.roi_id == b.roi_id:
        return True
    if a.roi_id not in rois or b.roi_id not in rois:
        return False
    return rois[a.roi_id].overlaps(rois[b.roi_id])


def prioritize_and_deduplicate(
    proposals: list[TemporalProposal], rois: list[ROI], config: TemporalConfig
) -> list[TemporalProposal]:
    """Keep the preferred window; union provenance, without extending it into a long event.

    Compare every member of a group to prevent transitive overlap from swallowing
    unrelated episodes. Simultaneous actions in overlapping regions remain ambiguous.
    """
    lookup = {roi.roi_id: roi for roi in rois}
    ordered = sorted(
        proposals,
        key=lambda p: (
            -config.roi_priorities.get(p.roi_id or "", 0),
            -p.motion_score,
            p.peak_sec,
            p.roi_id or "",
        ),
    )
    groups: list[list[TemporalProposal]] = []
    for proposal in ordered:

        def same(other: TemporalProposal, proposal: TemporalProposal = proposal) -> bool:
            overlap = max(
                0,
                min(proposal.activity_end_sec, other.activity_end_sec)
                - max(proposal.activity_start_sec, other.activity_start_sec),
            )
            duration = min(
                proposal.activity_end_sec - proposal.activity_start_sec,
                other.activity_end_sec - other.activity_start_sec,
            )
            return overlap / max(duration, 1e-9) >= config.dedup_overlap and _spatial_overlap(
                proposal, other, lookup
            )

        group = next((g for g in groups if all(same(other) for other in g)), None)
        if group is None:
            groups.append([proposal])
        else:
            group.append(proposal)
    return [
        g[0].model_copy(
            update={
                "roi_ids": sorted({roi_id for p in g for roi_id in p.roi_ids}),
                "segmentation_insufficient": any(p.segmentation_insufficient for p in g),
            }
        )
        for g in groups
    ]
