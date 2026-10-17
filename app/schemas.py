"""Shared ML → Rule Engine contract; no legal evaluation lives here."""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Observation(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class CandidateEvent(ContractModel):
    """Visual candidate; observations are tri-state, paths refer to local JPEGs."""

    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    start_sec: Seconds
    end_sec: Seconds
    frame_paths: list[str]
    roi_id: str | None = None
    observations: dict[str, Observation]

    @model_validator(mode="after")
    def ordered_times(self) -> Self:
        if self.start_sec > self.end_sec:
            raise ValueError("start_sec must not exceed end_sec")
        return self


class VerificationState(StrEnum):
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class VerifiedEvent(ContractModel):
    """Rule Engine output contract, not a rule implementation."""

    event_id: str = Field(min_length=1)
    state: VerificationState
    rule_id: str | None = None
    evidence_timestamps: list[Seconds]
    explanation: str

    @model_validator(mode="after")
    def confirmed_requires_rule(self) -> Self:
        if self.state == VerificationState.CONFIRMED and not self.rule_id:
            raise ValueError("confirmed requires a rule_id from the Rule Engine")
        return self
