"""Synthetic signal, geometry and evidence tests; no inference or bank footage."""

import numpy as np
import pytest
from pydantic import ValidationError

from app.vision.desk import DeskMotionAnalyzer
from app.vision.evidence import EventReplay, ReplayFrame, select_temporal_frames
from app.vision.motion import MotionConfig, MotionInterval
from app.vision.roi import ROI
from app.vision.temporal import (
    TemporalConfig,
    TemporalProposal,
    TransferBoundary,
    prioritize_and_deduplicate,
    propose_events,
)


def signal(scores, step=0.5, roi_id="desk"):
    return [
        MotionInterval(
            start_sec=i * step, end_sec=(i + 1) * step, motion_score=score, roi_id=roi_id
        )
        for i, score in enumerate(scores)
    ]


@pytest.mark.parametrize(
    "scores,count",
    [
        ([0] * 10, 0),
        ([0, 0, 0.08, 0.02, 0, 0], 1),
        ([0, 0.08, 0, 0, 0.09, 0, 0], 2),
        ([0.005] * 10, 0),
    ],
)
def test_hysteresis_pauses_and_no_motion(scores, count):
    batch = propose_events(signal(scores), TemporalConfig(), len(scores) * 0.5)
    assert len(batch.proposals) == count


def test_hysteresis_keeps_short_dip_and_splits_sufficient_pause():
    config = TemporalConfig(pause_sec=0.6)
    batch = propose_events(signal([0.08, 0.004, 0, 0.03, 0, 0, 0, 0.09], 0.25), config, 2)
    assert len(batch.proposals) == 2
    assert batch.proposals[0].activity_end_sec == 1


def test_continuous_motion_has_bounded_undersegmented_windows_not_one_long_triplet():
    batch = propose_events(signal([0.04] * 52), TemporalConfig(), 26)
    assert len(batch.proposals) > 1 and batch.continuous_regions == 1
    assert all(
        p.end_sec - p.start_sec <= 4 and p.segmentation_insufficient for p in batch.proposals
    )
    assert all(p.peak_sec in {s.end_sec for s in signal([0.04] * 52)} for p in batch.proposals)


def test_vfr_and_sparse_pts_preserved_and_proposal_limit_visible():
    samples = [
        MotionInterval(start_sec=a, end_sec=b, motion_score=0.1, roi_id="desk")
        for a, b in [(0, 0.17), (0.17, 0.81), (0.81, 7)]
    ]
    batch = propose_events(samples, TemporalConfig(max_proposals=1), 7)
    assert batch.truncated and batch.total > len(batch.proposals)
    assert batch.proposals[0].peak_sec in {0.17, 0.81, 7}
    for bad in [float("nan"), float("inf"), -1]:
        with pytest.raises(ValidationError):
            MotionInterval(start_sec=bad, end_sec=1, motion_score=0.1)
    duplicate = signal([0.1, 0.1]) + [signal([0.1])[0]]
    with pytest.raises(ValueError, match="nonmonotonic"):
        propose_events(duplicate, TemporalConfig(), 2)


def region(roi_id, left=0, right=1):
    return ROI(roi_id=roi_id, polygon=[(left, 0), (right, 0), (right, 1), (left, 1)])


def test_overlap_dedup_preserves_provenance_and_configurable_priority():
    config = TemporalConfig(roi_priorities={"custom": 7})
    samples = signal([0, 0.1, 0], roi_id="desk") + signal([0, 0.08, 0], roi_id="custom")
    batch = propose_events(samples, config, 1.5)
    result = prioritize_and_deduplicate(batch.proposals, [region("desk"), region("custom")], config)
    assert len(result) == 1 and result[0].roi_id == "custom"
    assert result[0].roi_ids == ["custom", "desk"]
    disjoint = prioritize_and_deduplicate(
        batch.proposals, [region("desk", 0, 0.3), region("custom", 0.7, 1)], config
    )
    assert len(disjoint) == 2


def test_desk_clips_overlapping_rois_and_excludes_external_motion():
    rois = [region("desk", 0, 0.4), region("cash_area"), region("transfer_zone")]
    analyzer = DeskMotionAnalyzer(MotionConfig(blur_size=1), rois)
    image = np.zeros((100, 200, 3), np.uint8)
    analyzer.update(image, 0)
    external = image.copy()
    external[30:70, 130:180] = 255
    assert all(s.motion_score == 0 for s in analyzer.update(external, 0.5))
    inside = external.copy()
    inside[30:70, 20:60] = 255
    scores = analyzer.update(inside, 1)
    assert len(scores) == 3 and all(s.motion_score > 0.01 for s in scores)


def test_boundary_geometry_is_finite_optional_and_not_a_transfer_observation():
    with pytest.raises(ValidationError):
        TransferBoundary(camera_profile="cam", points=((0.5, 0.5), (0.5, 0.5)))
    with pytest.raises(ValidationError):
        TransferBoundary(camera_profile="cam", points=((-1, 0.5), (0.5, 1)))
    boundary = TransferBoundary(enabled=True, camera_profile="cam", points=((0.5, 0.2), (0.5, 0.8)))
    analyzer = DeskMotionAnalyzer(MotionConfig(), [region("desk")], boundary=boundary)
    blank = np.zeros((100, 100, 3), np.uint8)
    changed = blank.copy()
    changed[40:60, 47:53] = 255
    analyzer.update(blank, 0)
    scores = analyzer.update(changed, 0.5)
    assert analyzer.boundary_activity and scores[0].motion_score > 0
    assert "object_transfer" not in scores[0].model_dump()
    inactive = DeskMotionAnalyzer(MotionConfig(), [], boundary=boundary)
    inactive.update(blank, 0)
    inactive.update(changed, 1)
    assert not inactive.boundary_activity


def evidence(t, fingerprint=None):
    return ReplayFrame(
        path=f"{t}.jpg",
        timestamp_sec=t,
        source_pts=round(t * 1000),
        time_base="1/1000",
        source_time_sec=t + 12,
        role="during",
        fingerprint=fingerprint or str(t),
    )


def test_temporal_roles_bracket_actual_peak_and_missing_roles_never_filled():
    p = TemporalProposal(
        start_sec=1,
        end_sec=4,
        activity_start_sec=2,
        activity_end_sec=3,
        peak_sec=2.5,
        motion_score=0.1,
    )
    selected, complete = select_temporal_frames([evidence(t) for t in [1, 2, 2.4, 2.6, 3, 3.5]], p)
    assert complete and [f.role for f in selected] == ["before", "during", "after"]
    assert selected[1].timestamp_sec == 2.4
    selected, complete = select_temporal_frames([evidence(t) for t in [2.4, 2.6, 3.5]], p)
    assert not complete and "before" not in [f.role for f in selected]
    same, complete = select_temporal_frames([evidence(t, "same") for t in [1, 2.5, 3.5]], p)
    assert not complete and len(same) == 1
    with pytest.raises(ValidationError):
        EventReplay(
            event_id="e", start_sec=1, end_sec=4, peak_sec=2.5, frames=same, temporal_complete=True
        )


@pytest.mark.parametrize(
    "update",
    [
        {"high_threshold": 0.01, "low_threshold": 0.02},
        {"refine_fps": float("inf")},
        {"pre_roll_sec": 3, "post_roll_sec": 3},
        {"pause_sec": 0},
        {"unknown": 1},
    ],
)
def test_config_rejects_invalid_boundaries(update):
    with pytest.raises(ValidationError):
        TemporalConfig(**update)


def test_peak_selection_uses_distinct_action_frame_and_measured_quiet_context():
    p = TemporalProposal(
        start_sec=0,
        end_sec=2,
        activity_start_sec=0.5,
        activity_end_sec=1.25,
        peak_sec=1.25,
        motion_score=0.1,
        before_sec=0.5,
        after_sec=1.5,
    )
    frames = [
        evidence(0.5, "before"),
        evidence(1, "moving"),
        evidence(1.25, "final"),
        evidence(1.5, "final"),
        evidence(2, "unrelated"),
    ]
    selected, complete = select_temporal_frames(frames, p)
    assert complete and [f.timestamp_sec for f in selected] == [0.5, 1, 1.5]


def test_roi_overlap_is_exact_for_thin_nested_and_concave_regions():
    a = region("a", 0.50001, 0.50002)
    b = region("b", 0.50003, 0.50004)
    assert not a.overlaps(b)
    assert region("wide").overlaps(a)
    assert a.overlaps(region("touch", 0.50002, 0.6))
    concave = ROI(
        roi_id="concave", polygon=[(0, 0), (1, 0), (1, 0.2), (0.2, 0.2), (0.2, 1), (0, 1)]
    )
    nested = ROI(roi_id="inside", polygon=[(0.4, 0.4), (0.8, 0.4), (0.8, 0.8), (0.4, 0.8)])
    assert not concave.overlaps(nested)
