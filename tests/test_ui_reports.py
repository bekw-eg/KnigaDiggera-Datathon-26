"""UI imports exercise the real shared models using synthetic metadata only."""

import json

import pytest

from app.ui.reports import MAX_REPORT_BYTES, ReportError, validate_report


@pytest.fixture
def report():
    return {
        "status": "partial",
        "events": [
            {
                "event_id": "synthetic-01",
                "event_type": "motion_candidate",
                "start_sec": 12.25,
                "end_sec": 15.5,
                "frame_paths": [],
                "observations": {"object_transfer": "unknown"},
            }
        ],
        "coverage": {"sample_fps": 2, "decoded_complete": False},
    }


def test_pipeline_report_preserves_timestamps_and_unknown(report):
    result = validate_report(json.dumps(report))
    event = result["analysis"]["events"][0]
    assert event["start_sec"] == 12.25
    assert event["observations"]["object_transfer"] == "unknown"
    assert result["verification"] == []


def test_optional_rule_result_keeps_legal_state_separate(report):
    bundle = {
        "analysis": report,
        "verification": [
            {
                "event_id": "synthetic-01",
                "state": "confirmed",
                "rule_id": "SYNTHETIC-TEST-ONLY",
                "evidence_timestamps": [12.25],
                "explanation": "Synthetic contract test, not a normative source.",
            }
        ],
    }
    result = validate_report(json.dumps(bundle))
    assert result["verification"][0]["state"] == "confirmed"
    assert "rule_id" not in result["analysis"]["events"][0]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_sec", -1),
        ("end_sec", float("nan")),
        ("end_sec", float("inf")),
        ("end_sec", 1),
        ("unexpected", "private content"),
    ],
)
def test_invalid_candidate_is_rejected_without_echoing_data(report, field, value):
    report["events"][0][field] = value
    with pytest.raises(ReportError) as error:
        validate_report(json.dumps(report))
    assert "private content" not in str(error.value)


def test_invalid_observation_is_not_repaired(report):
    report["events"][0]["observations"]["object_transfer"] = "probably"
    with pytest.raises(ReportError):
        validate_report(json.dumps(report))


@pytest.mark.parametrize("raw", ["{", "[]", "null", '{"analysis":{}}', None])
def test_malformed_or_missing_report(raw):
    with pytest.raises(ReportError):
        validate_report(raw)


def test_oversized_input_is_rejected():
    with pytest.raises(ReportError, match="8 MiB"):
        validate_report(" " * (MAX_REPORT_BYTES + 1))


@pytest.mark.parametrize("event_id", ["missing", "synthetic-01"])
def test_missing_rule_and_unlinked_verification_rejected(report, event_id):
    bundle = {
        "analysis": report,
        "verification": [
            {
                "event_id": event_id,
                "state": "confirmed",
                "evidence_timestamps": [],
                "explanation": "",
            }
        ],
    }
    with pytest.raises(ReportError):
        validate_report(json.dumps(bundle))


def test_duplicate_candidates_are_rejected(report):
    report["events"].append(report["events"][0].copy())
    with pytest.raises(ReportError):
        validate_report(json.dumps(report))


def test_technical_verified_does_not_supply_legal_confirmation(report):
    report["event_replay"] = [
        {
            "event_id": "synthetic-01",
            "start_sec": 12.25,
            "end_sec": 15.5,
            "peak_sec": 13,
            "status": "verified",
        }
    ]
    assert validate_report(json.dumps(report))["verification"] == []


def test_verification_timestamp_outside_candidate_is_rejected(report):
    bundle = {
        "analysis": report,
        "verification": [
            {
                "event_id": "synthetic-01",
                "state": "uncertain",
                "evidence_timestamps": [100],
                "explanation": "",
            }
        ],
    }
    with pytest.raises(ReportError):
        validate_report(json.dumps(bundle))
