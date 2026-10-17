"""Validate Stage 1 research integrity offline; never evaluate evidence or legal merits."""

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)

RESEARCH_DATE = date(2026, 10, 18)
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "docs" / "legal"
OBSERVATION_KEYS = frozenset(
    {
        "people_visible",
        "cash_visible",
        "phone_visible",
        "document_visible",
        "object_transfer",
        "reflection_possible",
        "physical_presence",
        "transferred_object_cash",
        "transferred_object_phone",
        "transferred_object_document",
        "transferred_object_other",
        "object_movement",
        "hand_object_interaction",
        "position_change_trackable",
        "object_occluded",
    }
)
OFFICIAL_HOSTS = frozenset(
    {
        "adilet.zan.kz",
        "old.adilet.zan.kz",
        "zan.gov.kz",
        "law.gov.kz",
        "nationalbank.kz",
        "www.nationalbank.kz",
        "gov.kz",
        "www.gov.kz",
    }
)
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _iso_date(value: object) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("expected ISO calendar date")
    return date.fromisoformat(value)


ISODate = Annotated[date, BeforeValidator(_iso_date)]


def _official_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in OFFICIAL_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("expected official HTTPS source")
    return value


class ResearchModel(BaseModel):
    """Strict research file shape, independent of shared event contracts."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Source(ResearchModel):
    """Official source metadata; null dates denote unresolved values."""

    source_id: NonEmpty
    title: NonEmpty
    authority: NonEmpty | None
    document_number: NonEmpty | None
    adoption_date: ISODate | None
    effective_date: ISODate | None
    revision_date: ISODate | None
    status: Literal["in_force", "unverified"]
    official_url: NonEmpty
    history_url: NonEmpty | None
    verified_at: ISODate
    notes: list[NonEmpty] = Field(min_length=1)

    @field_validator("official_url", "history_url")
    @classmethod
    def official_link(cls, value: str | None) -> str | None:
        return _official_url(value) if value is not None else None

    @model_validator(mode="after")
    def valid_metadata(self) -> Self:
        if self.verified_at > RESEARCH_DATE:
            raise ValueError("verification after research date")
        for value in (self.adoption_date, self.effective_date, self.revision_date):
            if value is not None and value > self.verified_at:
                raise ValueError("unverified future metadata date")
        if self.adoption_date and self.effective_date and self.effective_date < self.adoption_date:
            raise ValueError("effective date precedes adoption")
        for note in self.notes:
            for url in re.findall(r"https://[^\s]+", note):
                _official_url(url.rstrip(".,;"))
        return self


class RuleCandidate(ResearchModel):
    """Disabled research candidate, not an executable regulation."""

    candidate_rule_id: Annotated[str, StringConstraints(pattern=r"^LR-\d{3}$")]
    title: NonEmpty
    source_id: NonEmpty
    legal_reference: NonEmpty
    requirement_summary: NonEmpty
    applicability_conditions: list[NonEmpty] = Field(min_length=1)
    required_observations: list[NonEmpty]
    required_nonvisual_evidence: list[NonEmpty]
    video_verifiability: Literal["full", "partial", "none"]
    research_status: Literal["verified", "unverified", "requires_internal_policy", "needs_review"]
    limitations: list[NonEmpty] = Field(min_length=1)
    activation_eligible: StrictBool
    supporting_source_ids: list[NonEmpty]

    @model_validator(mode="after")
    def research_only(self) -> Self:
        if self.activation_eligible:
            raise ValueError("Stage 1 candidates must remain disabled")
        if not re.search(r"(?:п\.|пп\.|ст\.)\s*\d", self.legal_reference):
            raise ValueError("specific legal provision required")
        if set(self.required_observations) - OBSERVATION_KEYS:
            raise ValueError("unsupported pipeline observation")
        if len(set(self.required_observations)) != len(self.required_observations):
            raise ValueError("duplicate observations")
        if len(set(self.supporting_source_ids)) != len(self.supporting_source_ids):
            raise ValueError("duplicate supporting sources")
        if self.source_id in self.supporting_source_ids:
            raise ValueError("primary source repeated as supporting source")
        if self.video_verifiability == "full":
            if self.required_nonvisual_evidence or not self.required_observations:
                raise ValueError("full video verification cannot require nonvisual evidence")
        elif not self.required_nonvisual_evidence:
            raise ValueError("partial/none must identify missing evidence")
        return self


class ResearchValidationError(ValueError):
    """Malformed or internally inconsistent research, without network access."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ResearchValidationError("duplicate JSON key")
        output[key] = value
    return output


def _invalid_constant(value: str) -> None:
    raise ResearchValidationError("nonfinite JSON constant")


def _load(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_invalid_constant,
    )


def validate_research(
    directory: Path = DEFAULT_DIRECTORY,
) -> tuple[list[Source], list[RuleCandidate]]:
    """Load JSON and reject bad references, unsupported observations or activation.

    Checks structure and cross references only, not legal correctness, currency or
    applicability to a transaction. Raises ResearchValidationError on invalid data.
    """
    try:
        sources = TypeAdapter(list[Source]).validate_python(
            _load(directory / "source_registry.json")
        )
        candidates = TypeAdapter(list[RuleCandidate]).validate_python(
            _load(directory / "rule_candidates.json")
        )
        by_id = {source.source_id: source for source in sources}
        if len(by_id) != len(sources):
            raise ResearchValidationError("duplicate source ID")
        if len({rule.candidate_rule_id for rule in candidates}) != len(candidates):
            raise ResearchValidationError("duplicate candidate ID")
        for rule in candidates:
            ids = [rule.source_id, *rule.supporting_source_ids]
            if any(source_id not in by_id for source_id in ids):
                raise ResearchValidationError("unknown source ID")
            if rule.research_status in {"verified", "requires_internal_policy"}:
                if any(by_id[source_id].status != "in_force" for source_id in ids):
                    raise ResearchValidationError("unverified source promoted to verified basis")
                if any(
                    by_id[source_id].adoption_date is None
                    or by_id[source_id].effective_date is None
                    for source_id in ids
                ):
                    raise ResearchValidationError("verified basis lacks adoption/effective date")
        return sources, candidates
    except (OSError, ValueError) as exc:
        if isinstance(exc, ResearchValidationError):
            raise
        raise ResearchValidationError("invalid research files or field values") from exc


def main() -> int:
    """Run offline validation; return nonzero without dumping file contents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    try:
        sources, candidates = validate_research(args.directory)
    except ResearchValidationError as exc:
        print(f"legal_research_invalid: {exc}")
        return 1
    print(f"legal_research_valid: sources={len(sources)} candidates={len(candidates)} active=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
