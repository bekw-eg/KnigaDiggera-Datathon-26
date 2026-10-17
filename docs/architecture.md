# Shared contract v0.1 (initial foundation)

The full case specification v2.0 was not supplied separately. This implements the
field names and constraints explicitly supplied in the task; full v2.0 compatibility
must be reviewed by the team before integration. Subsequent contract changes need
all three participants' agreement.

```python
from app.schemas import CandidateEvent, Observation, VerifiedEvent
```

CandidateEvent has event_id, event_type, start_sec, end_sec, frame_paths, roi_id and
observations (dictionary of yes/no/unknown). Times are finite, nonnegative seconds,
start <= end. No Base64, legal verdict, probability or rule identifiers are emitted
by ML. Pydantic v2 `model_dump(mode="json")` and `model_dump_json()` serialize enums.

VerifiedEvent is a separate Rule Engine output: event_id, state (confirmed,
uncertain, rejected), optional rule_id, evidence_timestamps and explanation.
Confirmed requires a rule identifier; unconfirmed results may have rule_id=None.
ML does not construct legal decisions. Backend owns final report serialization,
evidence retention, async scheduling and persistence.
