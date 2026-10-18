"""Validate local review imports without reading paths or altering shared models."""

import json
from typing import Any, Self

from pydantic import Field, ValidationError, model_validator

from app.schemas import ContractModel, VerifiedEvent
from app.vision.result import VisionAnalysisResult

MAX_REPORT_BYTES = 8 * 1024 * 1024


class ReportError(ValueError):
    """Safe user-facing import error; excludes raw report content and local paths."""


class ReviewReport(ContractModel):
    """UI-only bundle: unchanged pipeline result and optional Rule Engine outputs.

    The pipeline's plain JSON is also accepted by :func:`validate_report`.
    Normative citations are deliberately absent: VerifiedEvent does not supply them.
    """

    analysis: VisionAnalysisResult
    verification: list[VerifiedEvent] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def check_links(self) -> Self:
        events = {event.event_id: event for event in self.analysis.events}
        if len(events) != len(self.analysis.events) or len(events) > 1000:
            raise ValueError("invalid event identifiers or count")
        for records in (self.analysis.event_replay, self.verification):
            ids = [record.event_id for record in records]
            if len(set(ids)) != len(ids) or any(event_id not in events for event_id in ids):
                raise ValueError("unlinked or duplicate event records")
        for record in self.verification:
            event = events[record.event_id]
            if any(not event.start_sec <= t <= event.end_sec for t in record.evidence_timestamps):
                raise ValueError("verification timestamp outside candidate")
        return self


def validate_report(raw: str) -> dict[str, Any]:
    """Return JSON-safe, Pydantic-validated data; reject malformed/oversized input.

    No coercive JSON repair, file access, inference, logging, or rule evaluation occurs.
    Error messages intentionally contain neither validation inputs nor evidence paths.
    """
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ReportError("Report must be a JSON file smaller than 8 MiB.")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("report must be an object")
        bundle = data if "analysis" in data else {"analysis": data}
        return ReviewReport.model_validate(bundle).model_dump(mode="json")
    except (ValueError, TypeError, ValidationError, RecursionError) as exc:
        raise ReportError(
            "Invalid report. Expected VisionAnalysisResult JSON or an "
            "analysis/verification bundle. "
            "Check fields, enums, finite timestamps, chronology and event links."
        ) from exc
