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

## Vision interface and timing

`from app.vision.pipeline import analyze_video, VisionConfig` provides the public
synchronous interface. `VisionAnalysisResult` (app/vision/result.py) is the ML
diagnostic envelope, not a modification of CandidateEvent. Backend may serialize
it separately from the public API's final report.

```text
local MOV -> PyAV stream + PTS -> sampled BGR -> motion intervals per calibrated ROI
 -> representative evidence JPEGs -> local Ollama schema -> ReflectionGuard
 -> CandidateEvent[] -> participant 3 Rule Engine -> VerifiedEvent[]
```

Timestamps are seconds relative to stream.start_time*time_base. If stream origin
is missing, first valid frame PTS supplies the origin; original source_pts,
time_base and source_time_sec remain in evidence_frames. Missing, negative relative,
duplicate or decreasing PTS explicitly fail; no FPS fallback is used. VFR sampling
uses time thresholds, not frame index. Stream duration/FPS may be unavailable;
container/audio duration is not silently substituted for video duration. Decode
exhaustion is the authoritative complete flag.

Sampling is single-pass and streaming. Pixels for only one decoded sample are
kept during motion analysis; bounded JPEG paths, motion hits and timestamps remain
in memory. Local JPEG storage is bounded by max_sampled_frames per job, not a
global disk quota. Backend must implement retention/quota. Source/video rotation
metadata and multi-camera scene changes are not auto-calibrated.

## Candidate semantics

`event_type="motion_activity"` says only that a motion interval was proposed.
`roi_id` identifies the proposal region, not a proven object location. If active,
the same ROI is cropped with surrounding context for VLM; full sampled frames
remain the evidence. Close intervals merge with pre/post-roll. Long merged
intervals get at most three representative frames; short or interior actions can
be missed. Overlapping calibrated ROIs may yield overlapping candidates.

Observation keys are people_visible (person-like figure, possibly reflected),
cash_visible, phone_visible, document_visible, object_transfer, reflection_possible,
physical_presence and transferred_object_{cash,phone,document,other}. Every value
is yes/no/unknown. A categorical transferred_object from the private VLM schema
is mapped into these existing dictionary fields; unsupported identities stay
unknown. No new fields were added to the shared contract. Free-text model
observations are validated but deliberately not propagated or logged.

One-frame inference forces transfer/object identity unknown. Three-frame inference
is configurable and chronological, but provides only a preliminary temporal
observation, not proof of all intervening actions. Matching clothes/color/proximity
is never proof of identity, presence, transfer or reflection. ReflectionGuard
uses person-box/glass overlap when optional YOLO exists, or a glass ROI proposal;
without sufficient geometry it keeps presence unknown. No legal verification or
untrained confidence calibration is performed.

## Budgets, failures and parallel backend jobs

VisionMetrics records decoded/sampled frames, total candidates before caps, VLM
attempts (including readiness/failed operations), successful calls, decode/CV/JPEG
I/O/VLM/total durations. VLM time includes lock waiting and preparation. Timings
are exclusive phase measurements where possible; total also includes orchestration
and process startup/cleanup, so phase sums need not equal total.

Coverage separates decoded timeline, sampling rate and successful VLM intervals.
This is not a percentage of actions understood. max_sampled_frames, max_candidates,
max_vlm_calls and error diagnostics cause partial state and unknown observations.
A successful scan without motion emits no candidates and does not need Ollama.

The public supervisor uses Windows spawn, receives bounded checkpoints and
terminates a worker at the wall-clock deadline, including native decoder/SDK
stalls. Up to about two seconds of cleanup can follow. Checkpoints are not
database persistence. A backend worker must allow child processes; avoid a daemon
multiprocessing pool. If using async API, schedule in the participant 1 worker,
not directly on the event loop. Coordinate a global job queue for production.

An OS file lock in the local system temp directory plus a threading lock serializes
heavy VLM requests across local jobs and benchmark. Lock waits consume request and
job budgets. Closing/terminating the client does not prove immediate cancellation
inside Ollama. The backend should avoid overlapping repeated timed-out jobs.

## References used for implementation

- [PyAV time and PTS](https://pyav.org/docs/stable/api/time.html)
- [Ollama Python SDK](https://github.com/ollama/ollama-python)
- [Ollama vision image input](https://github.com/ollama/ollama/blob/main/docs/capabilities/vision.mdx)
- [Ollama structured outputs](https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx)

Documentation explains API shape; local tests and benchmarks determine actual
capabilities/performance. No legal sources were invented or evaluated in this module.
