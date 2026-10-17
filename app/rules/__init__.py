"""Rule Engine package: evaluation of CandidateEvents against RK banking regulations."""

from app.rules.engine import RULE_CATALOG, evaluate_candidate, evaluate_candidates

__all__ = ["RULE_CATALOG", "evaluate_candidate", "evaluate_candidates"]
