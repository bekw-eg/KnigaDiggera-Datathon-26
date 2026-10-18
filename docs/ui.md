# Banking Intelligence workspace

The existing `app/ui/main.py` Streamlit entry point hosts a single, bundled ES-module
component in `app/ui/frontend/`. This replaces the old UI; there is no second React
server, cloud hosting, CDN request, external font request or new backend route.
Inter and Lucide assets and their licenses are bundled locally. The frontend has no
Node dependencies or compilation step. Node 22+ is used only for checks/tests.

## Run locally

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,backend,ui]"
.\.venv\Scripts\python.exe -m streamlit run app/ui/main.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. Dark theme and Video Analysis are the defaults. Overview,
Events, Evidence, Legal Verification, Reports and Settings share the same workspace.
The theme is the only preference written to browser storage. API keys, reports,
recordings and source frames are session-only; reloading clears them. Browser
object URLs are revoked on replacement/unload. Streamlit report validation stays
on the local server. Do not expose the Streamlit port to a public network.

## Three explicit data sources

**Demo workspace:** four isolated synthetic metadata records, five candidates,
zero confirmed legal violations. No demo video, generated evidence images, real
inference, processing animation or performance claim. Every screen carries the
Demo Mode banner. Attaching a preview does not start inference or replace the demo
metadata. Source identity is explicitly unverified.

**Local pipeline report:** use Import report with the JSON already written by:

```powershell
.\.venv\Scripts\python.exe -m scripts.test_vision "D:\local-videos\sample.mov" --config configs/vision.temporal.example.json --report artifacts/review.json
```

Use only a local file and your actual calibrated camera/profile when required.
The command needs an installed local Ollama vision model; this UI does not download
models. `--no-vlm` allows a CV-only run, with the pipeline's actual limitations.
Real footage/model integration is opt-in and is not a prerequisite for the demo.

`app/ui/reports.py` validates JSON with the unchanged VisionAnalysisResult and
VerifiedEvent models. It accepts the pipeline JSON directly, or this UI-only bundle:

```json
{
  "analysis": "replace this string with the actual VisionAnalysisResult object",
  "verification": []
}
```

The `verification` array may contain actual Rule Engine VerifiedEvent objects,
linked to candidates by `event_id`. Do not fabricate them. Import rejects extra
fields, malformed JSON, invalid enums, non-finite/negative timestamps, duplicate or
unlinked event records and verification timestamps outside the candidate window.
Maximum report size: 8 MiB / 1000 events. Errors never echo raw report data.

Open local video attaches the recording for browser preview only. Attach frames
selects original JPEG/PNG files separately (up to 24 MiB each). Replay frames retain
their actual role, timestamp, source PTS and time base. Single-frame evidence without
a supplied role appears separately; the UI never guesses Before/During/After.
Filename matching requires a unique match. Verify source identity manually; the UI
does not attest file provenance or hash equivalence. Missing/ambiguous images remain
placeholders. Images can be inspected in a zoom dialog with Fit / 100% original pixels.
No report-supplied filesystem path is opened by Python or fetched by the browser.

**Local API:** start the existing backend separately:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In Settings enter the loopback HTTP origin and your configured X-API-Key. The UI
does not supply, create or persist credentials. It rejects non-loopback origins,
HTTPS, embedded credentials, query strings, fragments and non-root URL paths.
Requests reject redirects. Existing CORS support is used unchanged.

| Existing endpoint | UI use |
| --- | --- |
| `GET /health` | API health and reported queue size; never proof Ollama is ready |
| `GET /api/v1/jobs` | Job summaries (latest 50, as limited by backend) |
| `GET /api/v1/jobs/{job_id}` | Selected/active job detail and actual status |
| `POST /api/v1/jobs/upload` | Explicit Start Analysis, multipart file upload |

Only .MOV/.MP4, non-empty, up to 512 MiB are accepted in this UI. File selection
never uploads automatically. The upload bar uses actual transferred bytes; the
analysis pipeline never shows invented percentages. HTTP/auth/timeout/invalid
response errors remain visible, with manual Retry. Upload failures may leave an
accepted server job: refresh jobs before resubmitting. Polling runs every 2 seconds
for active jobs and stops on failure/terminal status or after 360 seconds. Stopping
UI monitoring does **not** cancel the backend; there is no cancellation endpoint.
Completed job detail with a missing/invalid verdict or result is an explicit error,
not a successful empty analysis. Summary-only list records are kept distinct.

## Contract gaps and interpretation

- The legacy job API returns only `verdict`, `violations` and evidence Base64, not
  CandidateEvent/VerifiedEvent, all candidates, replay, per-stage telemetry, actual
  processing measurements or pipeline coverage. It also collapses vision result
  completeness in the worker. The UI cannot reconstruct this information.
  API `done` is displayed as a completed **job**, with coverage unavailable.
- Legacy violation signals lack `rule_id` and a VerifiedEvent. They remain uncertain
  review signals, with a generated **UI row identifier**, not a fabricated candidate
  ID. No legacy `no_violation` result becomes a claim of compliance.
- Imported pipeline `partial`, `failed`, `timed_out` and `complete` are preserved.
  Actual `metrics.total_sec` is shown when supplied; otherwise time is unavailable.
  The configured backend budget (default 300 seconds) is not a throughput promise.
- Per-stage states are unknown unless explicitly present in the isolated demo test
  fixture. Counts/timings never imply a stage succeeded. Coverage displays supplied
  decoding/sampling/VLM intervals, never action recall or violation probability.
- Technical Replay `verified` never implies legal `confirmed`. Confirmation requires
  an actual supplied Rule Engine `state=confirmed` and `rule_id`. The UI applies no
  legal rules. The shared contracts still do not contain a verified official source,
  exact clause or applicability conditions, so the page says
  **«Нормативное основание не установлено»** and provides no invented official link.
  Export is unavailable because no backend export API exists.
- ROI overlay is explicitly disabled: neither current API nor pipeline report
  establishes a calibrated geometry **and matching camera identity**. The control
  explains the gap. Replay crop geometry is not silently treated as calibration.
  A future adapter can enable the existing control when those facts are supplied.
- MOV is a container; browser codec support varies. Playback errors are visible.
  Browser seeks use seconds, never frame index/FPS. Precise evidence uses the
  pipeline's source PTS; browser playback seeking itself is not an exact PTS audit.
  Full screen is enabled only when the embedding browser exposes that capability.
- No face identification, embeddings, remote model calls or media uploads beyond
  the explicitly configured loopback backend. ML, shared contracts, Rule Engine,
  API, worker and database implementations are untouched by this redesign.

## Components, checks and packaging

`components.js` provides AppShell, Sidebar, Header, StatCard, VideoPlayer,
AnalysisProgress, EventTimeline, EventsTable, EvidenceViewer, VerificationDetails,
StatusBadge, EmptyState and ErrorState. `data.js` contains pure transforms and
isolated demo fixtures; `api.js` is the small live adapter; `app.js` owns interactions,
media lifecycle, polling and the Streamlit report-validation bridge. All dynamic
text is HTML-escaped. Only local, blob and validated JPEG data URLs render media.

The layout supports the requested desktop widths and a collapsible mobile sidebar;
event tables become cards on mobile. Buttons, focus states, dialogs, empty/error
states, byte progress, loading indicator and toasts have semantic labels. Native
dialogs manage focus and Escape. Animations respect `prefers-reduced-motion`.

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
node --test tests/ui.test.mjs
npm --prefix app/ui/frontend run check
.\.venv\Scripts\python.exe -m pip wheel . --no-deps --wheel-dir artifacts/ui-dist
```

The wheel includes HTML, JavaScript, CSS, fonts, icons and licenses via setuptools
package-data. There is no frontend build output or `node_modules` to commit. Keep
synthetic browser test media/screenshots in ignored `artifacts/`, never PR attachments
containing bank footage. Automated model/video integration remains explicitly opt-in.

## Verification record (2026-10-18)

- Python: 226 passed, 2 opt-in integration tests skipped. The unchanged two-second
  supervisor test failed once during concurrent setup; its isolated rerun and the
  subsequent full run passed. No ML code or deadline test was changed.
- JavaScript: 11 tests passed, including loopback rejection, invalid API results,
  source timestamps, reflection/unknown filtering, legal/technical separation,
  unique frame matching and the Streamlit iframe remount regression.
- Ruff lint/format, JavaScript syntax checks and wheel build passed. Wheel contents
  were inspected for the component, font, icons and licenses.
- Browser: all seven pages checked at 1920, 1440, 1280, 1024, 768 and 390 pixels
  (42 combinations); no horizontal page overflow. Both themes, mobile navigation,
  filters, timestamp selection, missing frames and normative-source empty state
  were checked. No browser console errors observed.
- A disposable synthetic MOV and three source PNGs exercised report import,
  original-frame zoom, temporal seek and actual browser playback. Invalid JSON
  produced the expected visible error. A separate isolated instance of the real
  API/routes/SQLite, with a synthetic ML stub, exercised rejected credentials,
  explicit multipart upload and actual queued/processing/done polling.
- No real Ollama inference, banking-video quality evaluation, throughput benchmark
  or real camera ROI calibration was run. No artificial evidence is bundled.
