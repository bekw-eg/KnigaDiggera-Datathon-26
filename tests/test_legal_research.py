import copy
import json
from pathlib import Path

import pytest

from app.vision.vlm import TemporalVLMObservation
from scripts.validate_legal_research import (
    DEFAULT_DIRECTORY,
    OBSERVATION_KEYS,
    ResearchValidationError,
    validate_research,
)


@pytest.fixture
def research_files(tmp_path):
    for name in ("source_registry.json", "rule_candidates.json"):
        (tmp_path / name).write_bytes((DEFAULT_DIRECTORY / name).read_bytes())
    return tmp_path


def mutate(directory: Path, filename: str, change) -> None:
    path = directory / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_catalog_and_matrix_are_consistent():
    sources, rules = validate_research()
    assert sources and rules
    assert not any(rule.activation_eligible for rule in rules)
    assert OBSERVATION_KEYS == set(TemporalVLMObservation.unknown().tri_state())
    matrix = (DEFAULT_DIRECTORY / "visual_observability_matrix.md").read_text(encoding="utf-8")
    for rule in rules:
        assert matrix.count(f"| {rule.candidate_rule_id}:") == 1
        row = next(line for line in matrix.splitlines() if f"| {rule.candidate_rule_id}:" in line)
        assert rule.video_verifiability in row
        assert rule.research_status in row


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation_eligible", True),
        ("activation_eligible", 0),
        ("activation_eligible", "false"),
        ("video_verifiability", "full"),
        ("video_verifiability", "maybe"),
        ("research_status", "approved"),
        ("legal_reference", "общий закон"),
        ("required_observations", ["cash_count"]),
        ("required_observations", ["cash_visible", "cash_visible"]),
        ("required_nonvisual_evidence", []),
        ("source_id", "missing"),
        ("supporting_source_ids", ["missing"]),
        ("supporting_source_ids", ["NB231"]),
        ("supporting_source_ids", ["NB71_2025", "NB71_2025"]),
        ("extra_verdict", "confirmed"),
        ("requirement_summary", ""),
    ],
)
def test_reject_invalid_candidate(research_files, field, value):
    mutate(research_files, "rule_candidates.json", lambda data: data[0].update({field: value}))
    with pytest.raises(ResearchValidationError):
        validate_research(research_files)


@pytest.mark.parametrize(
    "field,value",
    [
        ("official_url", "https://adilet.kz/ru/laws/test"),
        ("official_url", "https://adilet.zan.kz.evil.example/rus/docs/test"),
        ("official_url", "http://adilet.zan.kz/rus/docs/test"),
        ("official_url", "https://user:secret@adilet.zan.kz/rus/docs/test"),
        ("official_url", "https://adilet.zan.kz:8443/rus/docs/test"),
        ("history_url", "https://commercial.example/history"),
        ("notes", ["See https://unofficial.example/law"]),
        ("verified_at", "2026-10-19"),
        ("verified_at", "18.10.2026"),
        ("verified_at", 0),
        ("effective_date", "2010-01-01"),
        ("effective_date", "2027-01-01"),
        ("revision_date", "2026-02-30"),
        ("adoption_date", None),
        ("status", "active"),
        ("legal_verdict", "confirmed"),
    ],
)
def test_reject_invalid_source(research_files, field, value):
    mutate(research_files, "source_registry.json", lambda data: data[0].update({field: value}))
    with pytest.raises(ResearchValidationError):
        validate_research(research_files)


@pytest.mark.parametrize("filename", ["source_registry.json", "rule_candidates.json"])
def test_duplicate_ids(research_files, filename):
    mutate(research_files, filename, lambda data: data.append(copy.deepcopy(data[0])))
    with pytest.raises(ResearchValidationError):
        validate_research(research_files)


def test_cannot_promote_unverified_dependency(research_files):
    # LR-031 has an in-force primary source but an unresolved amending source.
    mutate(
        research_files,
        "rule_candidates.json",
        lambda data: data[-1].update(research_status="verified"),
    )
    with pytest.raises(ResearchValidationError, match="unverified source promoted"):
        validate_research(research_files)


@pytest.mark.parametrize("status", ["verified", "requires_internal_policy"])
def test_unverified_primary_source_is_not_confirmed_basis(research_files, status):
    mutate(
        research_files,
        "source_registry.json",
        lambda data: data[0].update(status="unverified"),
    )
    mutate(
        research_files,
        "rule_candidates.json",
        lambda data: data[0].update(research_status=status),
    )
    with pytest.raises(ResearchValidationError):
        validate_research(research_files)


@pytest.mark.parametrize(
    "content",
    [
        '[{"source_id":"a","source_id":"b"}]',
        "[NaN]",
        "[Infinity]",
        "[-Infinity]",
        "{",
        "{}",
        "null",
    ],
)
def test_reject_nonstandard_or_wrong_shape_json(research_files, content):
    (research_files / "source_registry.json").write_text(content, encoding="utf-8")
    with pytest.raises(ResearchValidationError):
        validate_research(research_files)


def test_missing_file_is_visible(research_files):
    (research_files / "source_registry.json").unlink()
    with pytest.raises(ResearchValidationError):
        validate_research(research_files)


def test_empty_candidate_list_is_allowed(research_files):
    (research_files / "rule_candidates.json").write_text("[]", encoding="utf-8")
    _, rules = validate_research(research_files)
    assert rules == []


def test_no_network_or_inference_required(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("offline validation attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    sources, rules = validate_research()
    assert len(sources) > 0 and len(rules) > 0
