# Project Overview

Datathon 26 / ForteBank Cash Discipline AI extracts visual observations from local
cash-desk videos for a separate Rule Engine. Three developers have 24 hours.
Python 3.11+, PyAV, OpenCV, Pydantic v2, pytest and local Ollama are the foundation.
Target hardware is Windows 11 / RTX 4050 Laptop / 6 GB; do not promise a 300-second
throughput target without measured video benchmarks. A local supervisor must stop
analysis on its configured deadline and report incomplete coverage.

`app/schemas.py` is the shared contract; `app/vision/` belongs to participant 2.
`app/api/`, `app/core/`, `app/ui/` belong to participant 1; `app/rules/` belongs to
participant 3. `configs/` holds camera calibration, `scripts/` local CLIs,
`tests/` synthetic/mocked checks, `docs/` integration and team decisions.

# Architecture Rules

- ML emits CandidateEvent, never a legal verdict, rule identifier or violation.
- Rule Engine consumes shared models, not prompt wording or raw VLM JSON.
- Backend calls `analyze_video(Path, VisionConfig)`; do not put CV in API routes.
- Keep public interfaces typed and documented. Do not build a monolithic main.py.
- Preserve CandidateEvent field names. Contract changes require agreement from
  all three participants; record proposals in docs/architecture.md first.
- Do not implement backend jobs, API keys, database, UI or legal rules here.

# Development Rules

- Type public functions; validate boundaries with Pydantic (reject extra fields,
  invalid enums, NaN, infinity and negative timestamps).
- Keep numerical processing separate from orchestration; use pure functions for
  timestamp conversion, geometry and interval merging when practical.
- Configure sampling, motion thresholds, ROI, timeouts and model budgets.
- Raise typed errors or return explicit partial/failed states; never hide errors
  as a successful empty event list. Never fake inference or benchmark results.
- Use logging for technical status codes/counts/timings only. Never log raw model
  responses, document text, image bytes/Base64, personal data, tokens or secrets.
- Add dependencies only when needed. Optional YOLO must import/load lazily and
  must not download weights automatically. No face recognition, embeddings or re-ID.

# AI Safety & Reliability

- Inference stays local. Only loopback Ollama endpoints and local model names;
  reject cloud models. Never upload bank footage or frames to remote services.
- No identification of people or faces. No invented facts or legal citations.
- Prefer unknown when evidence is ambiguous; motion/detector scores are measured
  signals, not calibrated probabilities of a violation.
- Validate every structured model response. Bad JSON/enum/truncation is an error,
  not a repaired successful response. Images must be chronological and limited.
- Glass may reflect a person. An overlapping figure cannot by itself establish
  physical presence; color, proximity and synchronized motion are not proof.
- Demo ROI is uncalibrated. Enable only a calibrated, matching camera profile.
- At most one heavy VLM operation across local pipeline/benchmark processes;
  preserve this guard when adding workers. Temperature 0 is not determinism.
- Sampling can miss brief actions; report actual coverage and candidate limits.

# Git Rules

- Before working inspect remote/status/branch/local identity and `gh auth status`
  plus `gh api user --jq .login`. Participant 2 expects bekw-eg; if another login
  is active, stop remote actions and ask the user. Never invent a missing remote.
- Commit using the actual authorized executor's account-linked name/email. Use
  their verified email or GitHub ID+login noreply address, configured locally.
  Never substitute another participant's author, committer, token or email.
- main is the stable integration branch. Use feature/backend-api (participant 1),
  feature/vision-pipeline (participant 2), feature/rules-qa (participant 3), and
  feature/<name> for separate work. Do not guess the other account owners' roles.
- Only bootstrap an actually empty remote on main when rights/rules permit;
  include minimal docs, package structure and shared contract, then branch.
  Otherwise branch from current main without rewriting it.
- Conventional Commits; small, coherent changes; run relevant checks first.
- Push your feature branch and open a PR as the actual authenticated account.
  Include behavior, limitations and checks. No force push to main, rewritten
  foreign history, deletion of teammates' branches or fabricated co-authorship.
- The designated integrator reviews/merges from their own account. Participant 2
  must not merge. Verify every push/PR operation before reporting success.
- Do not change global Git configuration. Never commit confidential media,
  evidence JPEGs, model weights, datasets, secrets, environments or caches.

# Testing Rules

- Unit tests use synthetic images/videos and a mock Ollama SDK, no bank footage.
- Cover PTS (nonzero origin, VFR, missing/nonmonotonic), corrupt/missing/no-stream
  files, ROI polygons/masks/crops/calibration, motion and merged intervals.
- Cover connection errors/timeouts, model capability checks, invalid JSON/enums,
  uncertain objects, glass reflections, contract serialization and no legal verdict.
- Test budget/deadline termination and partial coverage, not just happy paths.
- Real model/MOV integration is opt-in with local paths and working Ollama.
- Run `python -m pytest`, `ruff check .`, `ruff format --check .` before PR.
  Never claim unrun checks, real inference or model quality evaluation.

# Completion Rules

Do not declare completion until code exists, critical tests pass, public
interfaces and limitations are documented, errors are visible, Git operations
are checked and a short factual report lists commits/branch/PR and unrun checks.
If local model or footage is missing, document the exact gap and installation or
run command; do not download a large model to make a report appear complete.
