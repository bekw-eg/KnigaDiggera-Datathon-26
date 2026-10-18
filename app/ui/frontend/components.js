import {
  escapeHTML as h,
  timestamp,
  title,
  legalState,
  summarize,
  STAGES,
  filteredEvents,
} from "./data.js";

export const icon = (name, cls = "") =>
  `<span aria-hidden="true" class="icon ${cls}" style="--icon:url('assets/${name}.svg')"></span>`;
export const button = (
  label,
  action,
  name = "",
  cls = "secondary",
  attrs = "",
) =>
  `<button class="btn ${cls}" data-action="${action}" ${attrs}>${name ? icon(name) : ""}${label}</button>`;

const STATUS = {
  complete: ["Complete", "teal"],
  partial: ["Partial", "amber"],
  failed: ["Failed", "red"],
  timed_out: ["Timed out", "red"],
  processing: ["Processing", "teal"],
  queued: ["Queued", "neutral"],
  confirmed: ["Confirmed", "red"],
  uncertain: ["Uncertain", "amber"],
  rejected: ["Rejected", "neutral"],
  verified: ["Technical: verified", "teal"],
  incomplete: ["Incomplete", "amber"],
  pending: ["Pending", "neutral"],
  skipped_budget: ["Budget limit", "amber"],
  unavailable: ["Unavailable", "neutral"],
};
export function StatusBadge(state) {
  const [label, color] = STATUS[state] || [state || "Unavailable", "neutral"];
  return `<span class="badge ${color}"><i></i>${h(label)}</span>`;
}

export function EmptyState(heading, text, name = "folder-open", action = "") {
  return `<div class="empty-state">${icon(name)}<h3>${h(heading)}</h3><p>${h(text)}</p>${action}</div>`;
}

export function ErrorState(message, canRetry = false) {
  return `<div class="error-state" role="alert">${icon("circle-alert")}<div><strong>Review needs attention</strong><p>${h(message)}</p></div>${canRetry ? button("Retry", "refresh", "refresh-cw") : ""}${button("Dismiss", "dismiss-error", "", "text-button")}</div>`;
}

const NAV = [
  ["overview", "Overview", "layout-dashboard"],
  ["analysis", "Video Analysis", "scan-line"],
  ["events", "Events", "list-filter"],
  ["evidence", "Evidence", "images"],
  ["legal", "Legal Verification", "scale"],
  ["reports", "Reports", "chart-no-axes-combined"],
  ["settings", "Settings", "settings-2"],
];

export function Sidebar(s) {
  return `<aside id="sidebar" class="sidebar ${s.mobileNav ? "is-open" : ""}" aria-label="Main navigation">
    ${button("", "menu", "x", "icon-button sidebar-close", 'aria-label="Close navigation menu"')}
    <a class="brand" href="#analysis" data-page="analysis"><span class="brand-symbol">${icon("scan-line")}</span><span>banking<span class="brand-light">intelligence</span></span></a>
    <div class="workspace-switch">${icon("layers")}<div><strong>Datathon 26</strong><span>Cash discipline workspace</span></div><span class="workspace-code">26</span></div>
    <span class="nav-label">WORKSPACE</span>
    <nav>${NAV.map(([id, label, name], i) => `${i === 5 ? '<div class="nav-divider"></div>' : ""}<a href="#${id}" data-page="${id}" class="nav-link ${s.page === id ? "active" : ""}" ${s.page === id ? 'aria-current="page"' : ""}>${icon(name)}<span>${label}</span>${id === "events" && s.current?.events.length ? `<span class="nav-count">${s.current.events.length}</span>` : ""}</a>`).join("")}</nav>
    <div class="sidebar-bottom"><div class="local-card">${icon("hard-drive")}<strong>Designed to stay local</strong><p>Review footage in your own environment.</p><div><i class="status-dot"></i> ${s.mode === "demo" ? "Demo workspace" : "Local workspace"}</div></div>
    <div class="sidebar-footer"><span class="avatar">KD</span><div><strong>KnigaDiggera</strong><span>Datathon 26 · Team workspace</span></div></div></div>
  </aside>`;
}

export function Header(s) {
  const label = NAV.find((n) => n[0] === s.page)?.[1] || "Workspace";
  return `<header class="header"><div class="breadcrumb">${button("", "menu", "menu", "icon-button mobile-menu", `aria-label="Open navigation" aria-expanded="${s.mobileNav}" aria-controls="sidebar"`)}<span>Workspace</span>${icon("chevron-right")}<strong>${label}</strong></div>
    <div class="header-actions"><span class="connection-status"><i class="status-dot ${s.health ? "online" : ""}"></i>${s.health ? "Local API connected" : "Local AI · not checked"}</span><span class="mode-pill ${s.mode === "demo" ? "demo" : ""}">${s.mode === "demo" ? "DEMO MODE" : s.mode === "report" ? "LOCAL REPORT" : "LOCAL API"}</span>
    ${button("", "theme", s.theme === "dark" ? "sun" : "moon", "icon-button", 'aria-label="Toggle color theme"')}${button("", "help", "circle-help", "icon-button", 'aria-label="Workspace help"')}</div></header>`;
}

export function AppShell(s, body) {
  return `${Sidebar(s)}${s.mobileNav ? '<button class="nav-backdrop" data-action="menu" aria-label="Close navigation"></button>' : ""}<div class="main-shell" ${s.mobileNav ? "inert" : ""}>${Header(s)}<main id="content" tabindex="-1">
    ${s.mode === "demo" ? `<div class="demo-notice">${icon("info")}<span><strong>Demo workspace.</strong> Synthetic test metadata only. No bank footage, model inference or real evidence.</span><button data-page="settings">Connect local API ${icon("arrow-up-right")}</button></div>` : ""}
    ${s.mode === "api" ? '<div class="source-note">' + icon("info") + "<span>The current API exposes job status and legacy signals. Pipeline completeness, candidate coverage and legal sources are not exposed. Import a pipeline report for detailed review.</span></div>" : ""}
    ${s.error ? ErrorState(s.error, s.mode === "api") : ""}${s.loading ? '<div class="loading-line" role="status" aria-label="Loading local data"></div><div class="skeleton-grid" aria-hidden="true"><i></i><i></i><i></i></div>' : ""}
    ${body}<footer class="page-footer"><span>${icon("lock-keyhole")} Local review · Evidence before conclusions</span><span>KnigaDiggera <i></i> Datathon 26</span></footer></main></div>`;
}

export function PageHeading(eyebrow, heading, subtitle, actions = "") {
  return `<div class="page-heading"><div><p class="eyebrow">${eyebrow}</p><h1>${heading}</h1><p class="subtitle">${subtitle}</p></div><div class="page-actions">${actions}</div></div>`;
}

export function StatCard(label, value, name, note = "Loaded records") {
  return `<div class="stat-card"><div><span>${label}</span>${icon(name)}</div><strong>${value}</strong><small>${note}</small></div>`;
}

function AnalysisSelector(s) {
  return `<div class="analysis-selector">${icon("file-video")}<select id="analysis-select" aria-label="Selected analysis">${s.analyses.map((a) => `<option value="${h(a.id)}" ${s.current?.id === a.id ? "selected" : ""}>${h(a.name)}</option>`).join("")}</select></div>`;
}

export function Observations(event) {
  const entries = Object.entries(event?.observations || {});
  return entries.length
    ? `<dl class="observations">${entries.map(([key, value]) => `<div><dt>${h(title(key))}</dt><dd class="observation ${value}">${h(value)}</dd></div>`).join("")}</dl>`
    : '<p class="muted small">Structured observations are not supplied by this API.</p>';
}

export function EventTimeline(s) {
  const a = s.current,
    duration = s.timelineDuration;
  return `<div class="timeline"><input id="seek" type="range" min="0" max="${duration || 1}" step="0.01" value="${Math.min(s.time, duration || 1)}" ${!duration ? "disabled" : ""} aria-label="Video timestamp">
    <div class="timeline-markers">${
      duration
        ? a.events
            .filter((e) => e.start_sec <= duration)
            .map(
              (e) =>
                `<button class="event-marker ${s.selectedEvent === e.event_id ? "selected" : ""}" style="left:${(e.start_sec / duration) * 100}%" data-event="${h(e.event_id)}" aria-label="${h(title(e.event_type))} at ${timestamp(e.start_sec, true)}" title="${timestamp(e.start_sec, true)} · ${h(title(e.event_type))}"></button>`,
            )
            .join("")
        : ""
    }</div>
    <div class="timeline-scale"><span>00:00</span><span>${timestamp(duration ? duration / 4 : null)}</span><span>${timestamp(duration ? duration / 2 : null)}</span><span>${timestamp(duration ? (duration * 3) / 4 : null)}</span><span>${timestamp(duration)}</span></div></div>`;
}

export function VideoPlayer(s) {
  const a = s.current;
  return `<section class="video-panel panel" aria-label="Video player"><div class="panel-heading"><div class="panel-title">${icon("file-video")}<span>${h(a.name)}</span></div><span class="subtle-label">${a.source === "demo" ? "SYNTHETIC TEST" : "LOCAL SOURCE"}</span></div>
    <div class="video-stage ${s.videoUrl ? "has-video" : ""}" id="video-stage">
      ${s.videoUrl ? `<video id="source-video" src="${h(s.videoUrl)}" playsinline preload="metadata" aria-label="Local recording preview"></video><span class="video-source-note">Manual attachment · source identity unverified</span>` : `<div class="video-grid"></div><div class="corner top-left"></div><div class="corner top-right"></div><div class="corner bottom-left"></div><div class="corner bottom-right"></div><div class="video-placeholder"><div class="viewfinder">${icon("scan-line")}</div><h2>The full picture starts here.</h2><p>Attach a local recording to explore events in context.</p>${button("Open local video", "choose-video", "folder-open", "primary")}<small>.MOV / .MP4 · Preview stays in this browser</small></div>`}
      <div class="stage-label"><span class="record-dot"></span>${s.videoUrl ? "LOCAL PREVIEW" : "SOURCE NOT ATTACHED"}</div><span class="stage-time" data-time>${timestamp(s.time, true)}</span>
    </div>
    <div class="player-controls"><div class="control-group">${button("", "play", "play", "icon-button", `aria-label="Play or pause" ${!s.videoUrl ? "disabled" : ""}`)}${button("", "back", "skip-back", "icon-button", `aria-label="Back 5 seconds" ${!s.videoUrl ? "disabled" : ""}`)}${button("", "forward", "skip-forward", "icon-button", `aria-label="Forward 5 seconds" ${!s.videoUrl ? "disabled" : ""}`)}<span class="player-time"><strong data-time>${timestamp(s.time, true)}</strong><span data-duration>/ ${timestamp(s.timelineDuration)}</span></span></div><div class="control-group"><button class="roi-toggle" disabled title="No calibrated, matching camera geometry is supplied by the current report or API.">${icon("scan")} ROI <span class="toggle"></span></button>${button("", "fullscreen", "maximize", "icon-button", `aria-label="Full screen video" ${!s.fullscreenAvailable ? 'disabled title="Full screen is unavailable in this browser host."' : ""}`)}</div></div>
    ${EventTimeline(s)}<div class="player-caption"><span><i class="timeline-key"></i> Candidate event</span><span>${a.events.length} events <i>·</i> ${s.videoUrl ? "Browser playback" : "Attach source to play"}</span></div>
  </section>`;
}

function EventCard(e, s, i) {
  return `<button class="event-card ${s.selectedEvent === e.event_id ? "selected" : ""}" data-event="${h(e.event_id)}"><div class="event-card-top"><span class="event-index">${String(i + 1).padStart(2, "0")}</span><span class="mono">${timestamp(e.start_sec, true)}</span>${StatusBadge(legalState(e))}</div><h3>${h(title(e.event_type))}</h3><div class="event-card-bottom"><span>${icon("scan")} ${h(e.roi_id || "ROI unavailable")}</span>${icon("arrow-up-right")}</div></button>`;
}

export function EventList(s) {
  return `<section class="panel event-list"><div class="panel-heading"><div><h2>Detected events <span class="count">${s.current.events.length}</span></h2><p>Visual candidates for review</p></div>${icon("list-filter")}</div><div class="event-scroll">${s.current.events.length ? s.current.events.map((e, i) => EventCard(e, s, i)).join("") : EmptyState("No events supplied", "A result with no events does not establish compliance.", "scan-line")}</div><button class="panel-link" data-page="events">Open Events Explorer ${icon("arrow-right")}</button></section>`;
}

function getFrames(s) {
  const e = s.event;
  return ["before", "during", "after"].map((role) => ({
    role,
    frame: e?.replay?.frames?.find((f) => f.role === role),
  }));
}

export function EvidenceFrames(s, large = false) {
  return `<section class="evidence-section ${large ? "large" : ""}"><div class="section-heading"><div><h2>Evidence, in sequence</h2><p>Actual source timestamps. Original frames.</p></div>${button("Attach frames", "choose-frames", "images", "text-button", s.mode === "demo" ? 'disabled title="Switch to a real local report to attach evidence."' : "")}</div>
    <div class="frame-grid">${getFrames(s)
      .map(({ role, frame }, i) => {
        const src = frame ? s.frameURL(frame) : null;
        return `<button class="frame-card ${role === "during" ? "during" : ""}" data-role="${role}" ${!s.event ? "disabled" : ""}><div class="frame-header"><span><b>${String(i + 1).padStart(2, "0")}</b> ${title(role)}</span><span>${frame ? timestamp(frame.timestamp_sec, true) : "—"}</span></div><div class="frame-image">${src ? `<img src="${h(src)}" alt="${title(role)} source frame at ${timestamp(frame.timestamp_sec, true)}">${icon("zoom-in")}` : `${icon("images")}<span>${s.mode === "demo" ? "No demo image" : frame ? "Source image not attached" : "Frame not supplied"}</span>`}</div><div class="frame-footer">${src ? "Inspect original frame" : "Original evidence unavailable"}${icon(src ? "arrow-up-right" : "circle-help")}</div></button>`;
      })
      .join("")}</div>
    ${
      s.event?.legacyFrames?.length
        ? `<div class="legacy-frames"><h3>Source frames <span class="muted">· temporal roles not supplied</span></h3>${s.event.legacyFrames
            .map((f, i) => {
              const src = s.frameURL(f);
              return `<button class="legacy-frame" data-legacy="${i}">${src ? `<img src="${h(src)}" alt="Original source frame">` : icon("images")}<span>${timestamp(f.timestamp_sec, true)}</span></button>`;
            })
            .join("")}</div>`
        : ""
    }</section>`;
}

export function AnalysisProgress(s) {
  const a = s.current;
  return `<section class="panel pipeline"><div class="panel-heading"><div><h2>Analysis pipeline</h2><p>${a.stages ? "Demo stage metadata · no active processing" : "Stage telemetry is not supplied; individual steps remain unknown."}</p></div><div class="deadline"><span>${icon("clock-3")} ${s.budget} s budget</span><strong>${a.metrics?.total_sec != null ? `${a.metrics.total_sec.toFixed(1)} s measured` : "Processing time unavailable"}</strong></div></div>
    <ol class="pipeline-stages">${STAGES.map((stage, i) => {
      const status = a.stages?.[i];
      return `<li class="${status === "complete" ? "done" : status === "partial" ? "limited" : ""}"><span class="step-icon">${status === "complete" ? icon("check") : String(i + 1).padStart(2, "0")}</span><div><strong>${stage}</strong><span>${status ? STATUS[status]?.[0] || status : "Not reported"}</span></div></li>`;
    }).join("")}</ol>
    <div class="pipeline-note">${icon("info")} The deadline bounds analysis time; it does not guarantee full video coverage.</div></section>`;
}

export function Coverage(s) {
  const c = s.current.coverage;
  return `<div class="coverage-strip"><div>${icon("scan-line")}<span>Decoded timeline</span><strong>${c?.decoded_fraction != null ? `${(c.decoded_fraction * 100).toFixed(1)}%` : "Not supplied"}</strong></div><div>${icon("images")}<span>Sampled frames</span><strong>${s.current.metrics?.sampled_frames ?? "—"}</strong></div><div>${icon("activity")}<span>Sampling rate</span><strong>${c?.sample_fps != null ? `${c.sample_fps} fps` : "—"}</strong></div><div>${icon("cpu")}<span>VLM intervals</span><strong>${c?.vlm_intervals?.length ?? "—"}</strong></div></div>`;
}

export function VideoAnalysis(s) {
  const actions =
    button("Import report", "import-report", "download") +
    button("New analysis", "upload", "plus", "primary");
  const heading = PageHeading(
    "OBSERVE. UNDERSTAND. REVIEW.",
    "Video analysis",
    "Turn visual signals into a clear, reviewable sequence.",
    actions,
  );
  if (!s.current)
    return (
      heading +
      EmptyState(
        "Your review workspace is ready",
        "Choose a local recording or import an existing pipeline report.",
        "scan-line",
        button("Choose recording", "upload", "upload", "primary"),
      )
    );
  return `${heading}<div class="workspace-toolbar"><div class="toolbar-left">${AnalysisSelector(s)}${StatusBadge(s.current.status)}</div><div class="toolbar-meta">${icon("clock-3")} ${timestamp(s.current.duration)} duration<span class="toolbar-divider"></span>${s.current.source === "demo" ? "Test values" : "Local record"}</div></div>
    <div class="review-grid">${VideoPlayer(s)}${EventList(s)}</div>${EvidenceFrames(s)}${Coverage(s)}${AnalysisProgress(s)}
    <button class="dropzone-inline" data-action="upload">${icon("upload")}<div><strong>A new recording. A fresh perspective.</strong><span>Drop a .MOV or .MP4 file here, or browse your device.</span></div><span class="btn secondary">Choose recording ${icon("plus")}</span></button>`;
}

export function VerificationDetails(s) {
  const e = s.event,
    v = e?.verification;
  return `<section class="panel verification-details"><div class="panel-heading"><h2>Legal verification</h2>${StatusBadge(legalState(e))}</div><div class="detail-body">
    <div class="legal-empty">${icon("scale")}<h3>Нормативное основание не установлено</h3><p>No verified normative source or clause is supplied by the current contract.</p></div>
    <dl class="detail-list"><div><dt>Rule Engine state</dt><dd>${v ? h(v.state) : "Not supplied"}</dd></div><div><dt>Applied rule</dt><dd>${h(v?.rule_id || "Not supplied")}</dd></div><div><dt>Official source / clause</dt><dd>Not supplied</dd></div><div><dt>Applicability conditions</dt><dd>Not supplied</dd></div><div><dt>Missing evidence</dt><dd>${e?.frame_paths?.length ? "Check source frame availability" : "Source frames unavailable"}</dd></div></dl>
    <div class="detail-callout"><strong>Reasoning</strong><p>${h(v?.explanation || "Visual observations alone do not establish a legal conclusion. A Rule Engine result with an applicable rule is required.")}</p></div>
    ${e?.apiSignal ? '<p class="small muted">The legacy API reports a violation signal, but supplies no VerifiedEvent or rule_id. Its signal is retained for review and is not promoted to legally confirmed.</p>' : ""}</div></section>`;
}

export function EventDetails(s) {
  const e = s.event;
  if (!e)
    return `<section class="panel">${EmptyState("Select an event", "Event observations and diagnostics will appear here.", "list-filter")}</section>`;
  return `<section class="panel event-detail"><div class="panel-heading"><div><p class="eyebrow">EVENT DETAILS</p><h2>${h(e.event_id)}</h2></div>${StatusBadge(legalState(e))}</div><div class="detail-body"><h3>${h(title(e.event_type))}</h3><dl class="detail-list"><div><dt>Source timestamp</dt><dd class="mono">${timestamp(e.start_sec, true)} – ${timestamp(e.end_sec, true)}</dd></div><div><dt>Observation zone</dt><dd>${h(e.roi_id || "Not calibrated / unavailable")}</dd></div><div><dt>Evidence replay</dt><dd>${StatusBadge(e.replay?.status || "unavailable")}</dd></div></dl><h4>Visual observations</h4>${Observations(e)}
    <div class="detail-callout">${icon("info")}<p>Replay “verified” is a technical state. Legal confirmation comes from the Rule Engine.</p></div>
    <details><summary>Technical diagnostics ${icon("chevron-down")}</summary><div class="diagnostic-list">${(e.replay?.diagnostic_codes || []).map((c) => `<code>${h(c)}</code>`).join("") || '<span class="muted small">No diagnostics supplied.</span>'}</div></details>${button("Legal verification", "open-legal", "scale", "secondary full-width")}</div></section>`;
}

export function EvidenceViewer(s) {
  const heading = PageHeading(
    "THE CONTEXT BEHIND THE SIGNAL",
    "Evidence viewer",
    "Inspect the source. Keep observations and conclusions distinct.",
    button("Import report", "import-report", "download"),
  );
  if (!s.current)
    return (
      heading +
      EmptyState(
        "No source selected",
        "Import a local report to review its evidence.",
        "images",
      )
    );
  return `${heading}<div class="workspace-toolbar">${AnalysisSelector(s)}<div class="toolbar-left">${button("Previous", "prev-event", "arrow-left", "text-button", !s.event ? "disabled" : "")}${button("Next event", "next-event", "arrow-right", "text-button", !s.event ? "disabled" : "")}</div></div><div class="evidence-layout"><div>${VideoPlayer(s)}${EvidenceFrames(s, true)}<div class="source-note">${icon("info")} Attached images are selected manually. Verify them against the original recording and report.</div></div>${EventDetails(s)}</div>`;
}

export function EventsTable(s) {
  const events = s.current?.events || [],
    rows = filteredEvents(events, s.filter, s.query);
  return `${PageHeading("A SIGNAL IS THE START OF A REVIEW", "Events explorer", "Filter visual candidates by their separate legal verification state.", button("View evidence", "open-evidence", "images"))}
    ${s.current ? `<div class="workspace-toolbar">${AnalysisSelector(s)}<span class="muted small">${events.length} candidates in selected analysis</span></div>` : ""}
    <section class="panel events-explorer"><div class="explorer-tools"><div class="filter-tabs" role="group" aria-label="Verification filters">${[
      ["all", "All"],
      ["confirmed", "Confirmed"],
      ["uncertain", "Uncertain"],
      ["rejected", "Rejected"],
      ["unknown", "Unknown observations"],
    ]
      .map(
        ([id, label]) =>
          `<button data-filter="${id}" class="${s.filter === id ? "active" : ""}" aria-pressed="${s.filter === id}">${label}</button>`,
      )
      .join(
        "",
      )}</div><label class="search-field">${icon("search")}<input id="event-search" placeholder="Search events…" aria-label="Search events" value="${h(s.query)}"></label></div>
    <div class="events-table-wrap"><table class="events-table"><thead><tr><th>Event / type</th><th>Timestamp</th><th>ROI</th><th>Observations</th><th>Evidence</th><th>Legal state</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody>${rows.map((e) => `<tr class="${e.event_id === s.selectedEvent ? "selected" : ""}"><td data-label="Event"><strong>${h(e.event_id)}</strong><span>${h(title(e.event_type))}</span></td><td data-label="Timestamp" class="mono">${timestamp(e.start_sec, true)}</td><td data-label="ROI">${h(e.roi_id || "Unavailable")}</td><td data-label="Observations">${Object.values(e.observations).filter((v) => v === "unknown").length ? '<span class="unknown-label">Unknown present</span>' : Object.keys(e.observations).length ? "Available" : "Not supplied"}</td><td data-label="Evidence">${e.replay?.frames?.length || e.legacyFrames?.length || 0} frames</td><td data-label="Legal state">${StatusBadge(legalState(e))}</td><td>${button("Review", "review-event", "arrow-up-right", "text-button", `data-id="${h(e.event_id)}"`)}</td></tr>`).join("")}</tbody></table></div>
    ${!rows.length ? EmptyState("No matching events", "Try another filter or import a report with visual candidates.", "search") : ""}<div class="table-footer">${rows.length} of ${events.length} candidates <span>Detection does not establish a violation.</span></div></section>`;
}

export function LegalPage(s) {
  return `${PageHeading("SEPARATE OBSERVATION FROM JUDGEMENT", "Legal verification", "A legal result needs an applicable rule and sufficient evidence.")}${s.current ? `<div class="workspace-toolbar">${AnalysisSelector(s)}<select class="event-select" id="event-select" aria-label="Select event for legal review">${s.current.events.map((e) => `<option value="${h(e.event_id)}" ${e.event_id === s.selectedEvent ? "selected" : ""}>${h(e.event_id)} · ${timestamp(e.start_sec, true)}</option>`).join("")}</select></div>` : ""}${s.event ? `<div class="legal-layout">${EventDetails(s)}${VerificationDetails(s)}</div>` : EmptyState("No event to verify", "Choose an analysis that contains candidate events.", "scale")}`;
}

export function Overview(s) {
  const m = summarize(s.analyses),
    times =
      m.average != null
        ? `${m.average.toFixed(1)}<span class="stat-unit">s</span>`
        : "—";
  const dates = [
    ...new Set(
      s.analyses.filter((a) => a.created).map((a) => a.created.slice(0, 10)),
    ),
  ].sort();
  const counts = dates.map((d) =>
    s.analyses
      .filter((a) => a.created?.startsWith(d))
      .reduce((n, a) => n + a.events.length, 0),
  );
  const max = Math.max(...counts, 1),
    statuses = [
      ["Confirmed", m.confirmed, "red"],
      ["Uncertain", m.uncertain, "amber"],
      ["Rejected", m.rejected, "neutral"],
    ];
  const queue = s.analyses.filter((a) =>
    ["processing", "queued"].includes(a.status),
  );
  return `${PageHeading("YOUR LOCAL INTELLIGENCE WORKSPACE", "A clearer view of every review.", "A focused overview of the analyses loaded in this workspace.", button("New analysis", "upload", "plus", "primary"))}
    <div class="stats-grid">${StatCard("Total videos", m.total, "file-video")}${StatCard("Processed videos", m.processed, "circle-check", "Complete job / report records")}${StatCard("Detected events", m.events, "scan-line", "Visual candidates")}${StatCard("Confirmed violations", m.confirmed, "shield-check", "Rule Engine state + rule ID")}${StatCard("Uncertain events", m.uncertain, "circle-help", "Insufficient verification")}${StatCard("Avg. processing time", times, "clock-3", "Only supplied measurements")}</div>
    <div class="overview-charts"><section class="panel chart-panel"><div class="panel-heading"><div><h2>Event activity</h2><p>Candidate count by analysis date</p></div><span class="subtle-label">LOADED RECORDS</span></div><div class="activity-chart"><div class="chart-y"><span>${max}</span><span>${Math.round(max / 2)}</span><span>0</span></div><div class="bars">${dates.map((d, i) => `<div class="bar-column"><span class="bar-value">${counts[i]}</span><div class="bar" style="height:${(counts[i] / max) * 100}%" title="${d}: ${counts[i]} candidates"></div><span class="bar-label">${h(d.slice(5).replace("-", " / "))}</span></div>`).join("") || '<span class="muted">No activity data</span>'}</div></div></section>
    <section class="panel results-chart"><div class="panel-heading"><div><h2>Verification outcomes</h2><p>Legal states, separate from replay</p></div>${icon("scale")}</div><div class="donut-layout"><div class="donut" style="--confirmed:${m.events ? (m.confirmed / m.events) * 100 : 0}%;--uncertain:${m.events ? ((m.confirmed + m.uncertain) / m.events) * 100 : 0}%"><div><strong>${m.events}</strong><span>candidates</span></div></div><div class="chart-legend">${statuses.map(([label, value, color]) => `<div><span><i class="legend-dot ${color}"></i>${label}</span><strong>${value}</strong></div>`).join("")}</div></div></section></div>
    <div class="overview-bottom"><section class="panel recent"><div class="panel-heading"><div><h2>Recent analyses</h2><p>Pick up where the evidence left off</p></div>${button("Open workspace", "open-analysis", "arrow-up-right", "text-button")}</div>${s.analyses.map((a) => `<button class="recent-row" data-analysis="${h(a.id)}">${icon("file-video")}<div><strong>${h(a.name)}</strong><span>${h(a.created?.slice(0, 10) || "Date not supplied")} · ${a.events.length} candidates</span></div>${StatusBadge(a.status)}${icon("chevron-right")}</button>`).join("") || EmptyState("No analyses loaded", "Import a report or connect the local API.", "file-video")}</section>
    <div class="overview-aside"><section class="panel queue-panel"><div class="panel-heading"><h2>Processing queue</h2><span class="count">${queue.length}</span></div>${queue.length ? queue.map((a) => `<button class="recent-row" data-analysis="${h(a.id)}"><div><strong>${h(a.name)}</strong></div>${StatusBadge(a.status)}</button>`).join("") : '<div class="queue-empty">' + icon("circle-check") + "<div><strong>No queued jobs</strong><p>Nothing waiting in this workspace.</p></div></div>"}</section>
    <section class="panel system-panel"><div class="panel-heading"><h2>AI system status</h2>${icon("cpu")}</div><dl class="detail-list"><div><dt>Local API</dt><dd>${s.health ? "Connected" : "Not checked"}</dd></div><div><dt>Ollama / model readiness</dt><dd>Unknown</dd></div><div><dt>Inference concurrency</dt><dd>1 heavy VLM operation</dd></div><div><dt>Execution budget</dt><dd>${s.budget} s configured</dd></div></dl><button class="panel-link" data-page="settings">Connection settings ${icon("arrow-right")}</button></section></div></div>`;
}

export function Reports(s) {
  const a = s.current,
    m = summarize(a ? [a] : []);
  return `${PageHeading("A TRACEABLE RECORD OF THE REVIEW", "Analysis report", "Results, coverage and limitations in one place.")}${a ? `<div class="workspace-toolbar">${AnalysisSelector(s)}${StatusBadge(a.status)}</div><div class="report-summary panel"><div><p class="eyebrow">ANALYSIS SUMMARY</p><h2>${h(a.name)}</h2><p>${a.source === "demo" ? "Synthetic test values · no real inference or evidence" : "Local analysis record"}</p></div><div class="report-count"><strong>${m.events}</strong><span>detected events</span></div><div class="report-count"><strong>${m.confirmed}</strong><span>legally confirmed</span></div><div class="report-count"><strong>${m.uncertain}</strong><span>uncertain</span></div></div>${Coverage(s)}<div class="report-grid"><section class="panel"><div class="panel-heading"><h2>Coverage & processing</h2>${icon("activity")}</div><div class="detail-body"><dl class="detail-list"><div><dt>Processing time</dt><dd>${a.metrics?.total_sec != null ? `${a.metrics.total_sec.toFixed(2)} s` : "Not supplied"}</dd></div><div><dt>Decoded until</dt><dd>${timestamp(a.coverage?.decoded_until_sec, true)}</dd></div><div><dt>Decode complete</dt><dd>${a.coverage ? String(a.coverage.decoded_complete) : "Not supplied"}</dd></div><div><dt>Successful VLM intervals</dt><dd>${a.coverage?.vlm_intervals?.map(([b, e]) => `${timestamp(b, true)}–${timestamp(e, true)}`).join(", ") || "None supplied"}</dd></div></dl><div class="detail-callout"><p>Sampling can miss brief actions. Decoded coverage is not a measure of actions understood. No detected events does not mean no violations.</p></div></div></section><section class="panel"><div class="panel-heading"><h2>Technical diagnostics</h2>${icon("cpu")}</div><div class="detail-body diagnostic-list">${a.diagnostics.map((d) => `<code>${h(d.code)}</code>`).join("") || '<p class="muted">No diagnostic codes supplied.</p>'}<h4>Legal references</h4><p class="muted">Нормативное основание не установлено</p><p class="small muted">The current contracts do not provide verified normative citations.</p></div></section></div><div class="source-note">${icon("info")} Report export is unavailable: the backend has no export endpoint.</div>${button("Review detected events", "open-events", "arrow-right")}` : EmptyState("No report selected", "Import pipeline JSON or connect the local API to view results.", "chart-no-axes-combined")}`;
}

export function Settings(s) {
  return `${PageHeading("YOUR ENVIRONMENT, YOUR CONTROL", "Workspace settings", "Configure a local connection and choose how you review.")}
    <div class="settings-layout"><section class="panel"><div class="panel-heading"><div><h2>Data source</h2><p>Demo data is isolated from your local results.</p></div>${icon("plug")}</div><div class="detail-body"><div class="source-options"><button class="source-option ${s.mode === "demo" ? "selected" : ""}" data-action="demo"><span class="source-option-icon">${icon("layers")}</span><span><strong>Demo workspace</strong><small>Synthetic metadata · no inference</small></span>${s.mode === "demo" ? icon("circle-check") : ""}</button><button class="source-option ${s.mode === "report" ? "selected" : ""}" data-action="import-report"><span class="source-option-icon">${icon("file-video")}</span><span><strong>Local pipeline report</strong><small>Pydantic-validated JSON · no upload</small></span>${s.mode === "report" ? icon("circle-check") : ""}</button></div>
    <form id="connection-form"><h3>Connect existing local API</h3><label class="field-label" for="api-url">API origin</label><input id="api-url" class="text-input" type="url" required value="${h(s.endpoint)}" placeholder="http://127.0.0.1:8000"><label class="field-label" for="api-key">X-API-Key</label><input id="api-key" class="text-input" type="password" required autocomplete="off" value="" placeholder="Enter your locally configured API key"><p class="field-help">Loopback HTTP only. The key stays in memory and is never saved in browser storage.</p><button class="btn primary" type="submit" ${s.loading ? "disabled" : ""}>${icon("plug")} Connect & load jobs</button></form></div></section>
    <div><section class="panel"><div class="panel-heading"><h2>Appearance</h2>${icon("sun")}</div><div class="detail-body appearance"><span><strong>${s.theme === "dark" ? "Dark" : "Light"} theme</strong><p>Dark by default. Follows reduced-motion preferences.</p></span>${button("Switch theme", "theme", s.theme === "dark" ? "sun" : "moon")}</div></section><section class="panel integration-info"><div class="panel-heading"><h2>Integration availability</h2>${icon("info")}</div><dl class="detail-list"><div><dt>Upload / jobs / polling</dt><dd>Available in API</dd></div><div><dt>Stage telemetry / ROI geometry</dt><dd>Not exposed</dd></div><div><dt>CandidateEvent / EventReplay</dt><dd>Local report import</dd></div><div><dt>Rule Engine result</dt><dd>Optional report bundle</dd></div><div><dt>Verified source / report export</dt><dd>Not exposed</dd></div></dl></section></div></div>`;
}
