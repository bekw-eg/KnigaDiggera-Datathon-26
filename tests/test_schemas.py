import pytest
from pydantic import ValidationError

from app.schemas import CandidateEvent, VerifiedEvent


def candidate(**kwargs):
    return CandidateEvent(
        event_id="x",
        event_type="motion_activity",
        start_sec=0,
        end_sec=1,
        frame_paths=["local.jpg"],
        observations={"cash_visible": "unknown"},
        **kwargs,
    )


def test_roundtrip_and_no_legal_verdict():
    event = candidate()
    assert CandidateEvent.model_validate_json(event.model_dump_json()) == event
    assert event.observations["cash_visible"] == "unknown"
    assert "state" not in event.model_dump()
    with pytest.raises(ValidationError):
        candidate(state="confirmed")


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 2), (float("nan"), 2), (0, float("inf"))])
def test_invalid_times(start, end):
    with pytest.raises(ValidationError):
        CandidateEvent(
            event_id="x",
            event_type="x",
            start_sec=start,
            end_sec=end,
            frame_paths=[],
            observations={},
        )


def test_enum_and_verification_contract():
    with pytest.raises(ValidationError):
        CandidateEvent(
            event_id="x",
            event_type="x",
            start_sec=0,
            end_sec=1,
            frame_paths=[],
            observations={"cash_visible": "probably"},
        )
    event = VerifiedEvent(
        event_id="x", state="uncertain", evidence_timestamps=[0.5], explanation=""
    )
    assert event.rule_id is None
    with pytest.raises(ValidationError):
        VerifiedEvent(event_id="x", state="confirmed", evidence_timestamps=[], explanation="")
