import {
  demoAnalyses,
  normalizeReport,
  matchEvidence,
  timestamp,
  shouldAcceptReport,
  escapeHTML as h,
} from "./data.js";
import { LocalApiAdapter } from "./api.js";
import {
  AppShell,
  Overview,
  VideoAnalysis,
  EventsTable,
  EvidenceViewer,
  LegalPage,
  Reports,
  Settings,
  EventTimeline,
  icon,
  button,
} from "./components.js";

const $ = (selector) => document.querySelector(selector);
const embedded = window.parent !== window;
const views = {
  overview: Overview,
  analysis: VideoAnalysis,
  events: EventsTable,
  evidence: EvidenceViewer,
  legal: LegalPage,
  reports: Reports,
  settings: Settings,
};
let savedTheme = "dark";
try {
  savedTheme = localStorage.getItem("banking-theme") || "dark";
} catch {
  /* storage may be disabled */
}
const initial = location.hash.slice(1);
const state = {
  page: views[initial] ? initial : "analysis",
  theme: savedTheme === "light" ? "light" : "dark",
  mobileNav: false,
  mode: "demo",
  analyses: demoAnalyses(),
  selectedAnalysis: "demo-01",
  selectedEvent: "DEMO-001",
  time: 34.5,
  budget: 300,
  health: null,
  endpoint: "http://127.0.0.1:8000",
  adapter: null,
  error: null,
  loading: false,
  filter: "all",
  query: "",
  media: null,
  mediaDuration: null,
  files: [],
  fileURLs: new Map(),
  imageFailures: new Set(),
  uploadFile: null,
  uploading: false,
  uploadProgress: 0,
  requestId: null,
  lastResponseId: null,
  importName: "",
  pollTimer: null,
  pollStarted: null,
  bridgeReady: false,
};

function context() {
  const current =
    state.analyses.find((a) => a.id === state.selectedAnalysis) ||
    state.analyses[0];
  const event =
    current?.events.find((e) => e.event_id === state.selectedEvent) || null;
  const media = state.media?.analysisId === current?.id ? state.media : null;
  return {
    ...state,
    current,
    event,
    fullscreenAvailable: !!document.fullscreenEnabled,
    videoUrl: media?.url || null,
    timelineDuration:
      media && Number.isFinite(state.mediaDuration)
        ? state.mediaDuration
        : current?.duration || null,
    frameURL: (frame) => {
      if (state.mode === "demo") return null;
      const file = matchEvidence(frame.path, state.files);
      const url = frame.image || (file ? state.fileURLs.get(file) : null);
      return state.imageFailures.has(url) ? null : url;
    },
  };
}

function send(type, args = {}) {
  if (embedded)
    window.parent.postMessage({ isStreamlitMessage: true, type, ...args }, "*");
}
let heightFrame;
function updateHeight() {
  cancelAnimationFrame(heightFrame);
  heightFrame = requestAnimationFrame(() =>
    send("streamlit:setFrameHeight", { height: window.innerHeight }),
  );
}
new ResizeObserver(updateHeight).observe($("#app"));

function render() {
  document.documentElement.dataset.theme = state.theme;
  $("#app").innerHTML = AppShell(context(), views[state.page](context()));
  const video = $("#source-video");
  if (video) {
    video.addEventListener("loadedmetadata", () => {
      state.mediaDuration = Number.isFinite(video.duration)
        ? video.duration
        : null;
      const expected = context().current?.duration;
      if (
        expected &&
        state.mediaDuration &&
        Math.abs(expected - state.mediaDuration) > 1
      ) {
        toast(
          "Video duration differs from the report. Verify the source recording.",
        );
      }
      if (state.time < video.duration) video.currentTime = state.time;
      else updateTime(0);
      if (state.mediaDuration) {
        $(".timeline").outerHTML = EventTimeline(context());
        const duration = $("[data-duration]");
        if (duration)
          duration.textContent = `/ ${timestamp(state.mediaDuration)}`;
      }
    });
    video.addEventListener("timeupdate", () => updateTime(video.currentTime));
    video.addEventListener("play", () => {
      const b = $('[data-action="play"]');
      if (b) b.innerHTML = icon("pause");
    });
    video.addEventListener("pause", () => {
      const b = $('[data-action="play"]');
      if (b) b.innerHTML = icon("play");
    });
    video.addEventListener("error", () =>
      toast(
        "This video codec cannot be played by the browser. Local analysis accepts MOV independently of browser playback.",
      ),
    );
  }
  updateHeight();
}

let toastTimer;
function toast(message) {
  const box = $("#toast");
  box.textContent = message;
  box.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove("visible"), 5500);
}

function updateTime(time) {
  if (!Number.isFinite(time) || time < 0) return;
  state.time = time;
  document
    .querySelectorAll("[data-time]")
    .forEach((e) => (e.textContent = timestamp(time, true)));
  const seek = $("#seek");
  if (seek) seek.value = time;
}

function seekTo(time) {
  const video = $("#source-video");
  const target =
    video?.readyState >= 1 && Number.isFinite(video.duration)
      ? Math.min(time, video.duration)
      : time;
  if (target !== time)
    toast(
      "Selected timestamp is outside the attached recording. Verify the source identity.",
    );
  updateTime(target);
  if (video?.readyState >= 1) video.currentTime = target;
}

function navigate(page, focus = true) {
  if (!views[page]) return;
  state.page = page;
  state.mobileNav = false;
  location.hash = page;
  render();
  if (focus) $("#content").focus({ preventScroll: true });
}

function selectAnalysis(id, go = false) {
  const a = state.analyses.find((a) => a.id === id);
  if (!a) return;
  state.selectedAnalysis = id;
  state.selectedEvent = a.events[0]?.event_id || null;
  state.time = a.events[0]?.start_sec || 0;
  state.mediaDuration = null;
  state.filter = "all";
  state.query = "";
  if (go) navigate("analysis");
  else render();
}

function selectEvent(id, open = false) {
  const event = context().current?.events.find((e) => e.event_id === id);
  if (!event) return;
  state.selectedEvent = id;
  state.time = event.start_sec;
  if (open) navigate("evidence");
  else {
    render();
    seekTo(event.start_sec);
  }
}

function modal(title, body, cls = "") {
  const d = $("#dialog");
  d.className = cls;
  d.innerHTML = `<div class="dialog-header"><h2 id="dialog-title">${h(title)}</h2><div class="dialog-tools">${cls === "zoom-dialog" ? button("100%", "zoom-image", "zoom-in", "text-button") : ""}${button("", "close-dialog", "x", "icon-button", 'aria-label="Close dialog"')}</div></div><div class="dialog-body">${body}</div>`;
  if (!d.open) d.showModal();
}

function uploadDialog() {
  const f = state.uploadFile;
  modal(
    "A new perspective starts here.",
    `<div class="upload-zone" id="upload-zone">${icon("upload")}<strong>Drop your recording here</strong><p>.MOV or .MP4 · up to 512 MiB · local API only</p>${button("Browse device", "choose-video", "folder-open")}</div>
    ${f ? `<div class="file-selection">${icon("file-video")}<div><strong>${h(f.name)}</strong><span>${(f.size / 1024 / 1024).toFixed(2)} MiB · Ready for local preview</span></div>${icon("circle-check")}</div>` : ""}
    ${state.uploading ? `<div class="upload-progress"><div style="width:${state.uploadProgress * 100}%"></div></div><p id="upload-status">Uploading · ${Math.round(state.uploadProgress * 100)}% of bytes</p>` : ""}
    <p>${state.mode === "api" ? "Start Analysis sends this file to your configured loopback API. Model work runs on the local backend." : "Preview is available in this mode. To run analysis, connect the existing local API in Settings."}</p>
    <div class="dialog-actions">${button("Cancel", "close-dialog", "", "secondary", state.uploading ? "disabled" : "")}${state.mode === "api" ? button("Start Analysis", "start-analysis", "scan-line", "primary", !f || state.uploading ? "disabled" : "") : button("Connect local API", "connect-settings", "plug", "primary")}</div>`,
  );
}

function attachVideo(file) {
  if (!file || !/\.(mov|mp4)$/i.test(file.name) || file.size === 0) {
    toast("Choose a non-empty .MOV or .MP4 file.");
    return;
  }
  if (file.size > 512 * 1024 * 1024) {
    toast(
      "This UI accepts files up to 512 MiB. Use the local pipeline CLI for larger recordings.",
    );
    return;
  }
  if (state.uploading) {
    toast("Wait for the current upload to finish.");
    return;
  }
  if (state.media) URL.revokeObjectURL(state.media.url);
  state.uploadFile = file;
  state.mediaDuration = null;
  if (!context().current) {
    const draft = {
      id: `preview-${Date.now()}`,
      name: file.name,
      status: "pending",
      source: "preview",
      duration: null,
      events: [],
      metrics: null,
      coverage: null,
      diagnostics: [],
      created: null,
    };
    state.analyses.unshift(draft);
    state.selectedAnalysis = draft.id;
    state.selectedEvent = null;
  }
  state.media = {
    file,
    url: URL.createObjectURL(file),
    analysisId: context().current.id,
  };
  state.time = 0;
  render();
  if ($("#dialog").open) uploadDialog();
  toast("Recording attached for local preview. Analysis has not started.");
}

function clearAttachments() {
  if (state.media) URL.revokeObjectURL(state.media.url);
  state.fileURLs.forEach((url) => URL.revokeObjectURL(url));
  state.media = null;
  state.mediaDuration = null;
  state.uploadFile = null;
  state.files = [];
  state.fileURLs.clear();
  state.imageFailures.clear();
}

function resetSource(mode, analyses) {
  stopPolling();
  clearAttachments();
  state.mode = mode;
  state.analyses = analyses;
  state.selectedAnalysis = analyses[0]?.id || null;
  state.selectedEvent = analyses[0]?.events[0]?.event_id || null;
  state.time = analyses[0]?.events[0]?.start_sec || 0;
  state.error = null;
  state.query = "";
  state.filter = "all";
  if (mode !== "api") {
    state.adapter = null;
    state.health = null;
  }
}

async function connect(form) {
  if (state.uploading) return;
  const endpoint = form.querySelector("#api-url").value,
    key = form.querySelector("#api-key").value;
  if (!key.trim()) {
    toast("Enter the locally configured API key.");
    return;
  }
  state.loading = true;
  state.error = null;
  // Keep the in-progress form intact. Credentials are never persisted or logged.
  const submit = form.querySelector("button[type=submit]");
  submit.disabled = true;
  submit.textContent = "Connecting…";
  try {
    const adapter = new LocalApiAdapter(endpoint, key);
    const health = await adapter.health();
    const analyses = await adapter.list();
    resetSource("api", analyses);
    state.adapter = adapter;
    state.endpoint = adapter.endpoint;
    state.health = health;
    state.loading = false;
    navigate("analysis");
    toast(
      "Local API connected. Stage telemetry and model readiness are not exposed by this API.",
    );
    await refresh();
  } catch (error) {
    state.loading = false;
    state.error = error.message;
    render();
  }
}

function stopPolling() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  state.pollStarted = null;
}
function schedulePolling() {
  clearTimeout(state.pollTimer);
  if (
    state.mode !== "api" ||
    !state.analyses.some((a) => ["processing", "queued"].includes(a.status))
  )
    return;
  state.pollStarted ||= performance.now();
  if (performance.now() - state.pollStarted > 360000) {
    state.error =
      "UI monitoring stopped after 360 seconds. The backend may still be running; refresh jobs to check. No cancellation endpoint is available.";
    render();
    return;
  }
  state.pollTimer = setTimeout(() => refresh(false), 2000);
}

async function refresh(manual = true) {
  if (!state.adapter || state.mode !== "api") return;
  const adapter = state.adapter;
  if (manual) {
    state.loading = true;
    state.error = null;
    state.pollStarted = performance.now();
    render();
  }
  try {
    const list = await adapter.list();
    // Fetch full results only for the selected job and active jobs; the list is summary-only.
    const ids = list
      .filter(
        (a) =>
          a.id === state.selectedAnalysis ||
          ["queued", "processing"].includes(a.status),
      )
      .map((a) => a.id);
    const details = await Promise.all(ids.map((id) => adapter.get(id)));
    if (state.adapter !== adapter) return;
    const revision = (analyses) =>
      JSON.stringify(
        analyses.map((a) => [a.id, a.status, a.updated, a.events.length]),
      );
    const previousRevision = revision(state.analyses);
    const cache = new Map(state.analyses.map((a) => [a.id, a]));
    details.forEach((a) => {
      if (state.media?.analysisId === a.id) a.name = state.media.file.name;
      cache.set(a.id, a);
    });
    state.analyses = list.map((a) => {
      const full = cache.get(a.id);
      return full && full.status === a.status
        ? { ...full, name: full.name }
        : a;
    });
    if (!state.analyses.some((a) => a.id === state.selectedAnalysis))
      state.selectedAnalysis = state.analyses[0]?.id || null;
    const current = context().current;
    if (!current?.events.some((e) => e.event_id === state.selectedEvent))
      state.selectedEvent = current?.events[0]?.event_id || null;
    const needsRender =
      manual ||
      state.loading ||
      !!state.error ||
      previousRevision !== revision(state.analyses);
    state.error = null;
    state.loading = false;
    // Unchanged polls must not tear down an actively playing source video.
    if (needsRender) render();
    schedulePolling();
  } catch (error) {
    if (state.adapter !== adapter) return;
    clearTimeout(state.pollTimer);
    state.loading = false;
    state.error = error.message;
    render();
  }
}

async function startAnalysis() {
  if (
    state.mode !== "api" ||
    !state.adapter ||
    !state.uploadFile ||
    state.uploading
  )
    return;
  const adapter = state.adapter,
    file = state.uploadFile;
  state.uploading = true;
  state.uploadProgress = 0;
  uploadDialog();
  try {
    const job = await adapter.upload(file, (fraction) => {
      state.uploadProgress = fraction;
      const bar = $(".upload-progress>div"),
        text = $("#upload-status");
      if (bar) bar.style.width = `${fraction * 100}%`;
      if (text)
        text.textContent =
          fraction === 1
            ? "File transferred · waiting for job acceptance"
            : `Uploading · ${Math.round(fraction * 100)}% of bytes`;
    });
    job.name = file.name;
    state.analyses.unshift(job);
    state.selectedAnalysis = job.id;
    state.selectedEvent = null;
    if (state.media) state.media.analysisId = job.id;
    state.time = 0;
    state.uploadFile = null;
    state.uploading = false;
    state.pollStarted = performance.now();
    $("#dialog").close();
    navigate("analysis");
    toast("File accepted by the local API. Waiting for actual job status.");
    schedulePolling();
  } catch (error) {
    state.uploading = false;
    $("#dialog").close();
    state.error = error.message;
    render();
  }
}

async function importReport(file) {
  if (!file) return;
  if (
    !file.name.toLowerCase().endsWith(".json") ||
    file.size > 8 * 1024 * 1024
  ) {
    toast("Choose a JSON report smaller than 8 MiB.");
    return;
  }
  if (!embedded) {
    toast(
      "Validated report import is available through the Streamlit entry point.",
    );
    return;
  }
  const json = await file.text();
  state.requestId = crypto.randomUUID();
  state.importName = file.name;
  state.loading = true;
  render();
  send("streamlit:setComponentValue", {
    dataType: "json",
    value: { id: state.requestId, action: "validate_report", json },
  });
}

function showFrame(role) {
  const ctx = context(),
    frame = ctx.event?.replay?.frames?.find((f) => f.role === role);
  if (frame) seekTo(frame.timestamp_sec);
  const url = frame ? ctx.frameURL(frame) : null;
  if (!url) {
    toast(
      frame
        ? "Attach this original source image to inspect it."
        : "This temporal frame was not supplied. No image has been generated.",
    );
    return;
  }
  modal(
    `${role[0].toUpperCase() + role.slice(1)} · ${timestamp(frame.timestamp_sec, true)}`,
    `<img src="${h(url)}" alt="Original ${h(role)} frame"><div class="zoom-caption">Original manually attached image · source PTS ${h(frame.source_pts)} · time base ${h(frame.time_base)}</div>`,
    "zoom-dialog",
  );
}

function help() {
  modal(
    "A workspace built for review.",
    `<p>Start with Video Analysis. Open a local recording for preview, select an event to seek to its timestamp, then inspect the available source frames.</p><h3>Demo Mode</h3><p>All numbers are synthetic test metadata. There is no recording, inference or artificial evidence.</p><h3>Local reports</h3><p>Import the JSON emitted by scripts.test_vision. To include legal results, use an object with <code>analysis</code> (VisionAnalysisResult) and <code>verification</code> (VerifiedEvent array). Source images must be selected separately.</p><h3>Keep the distinction clear</h3><p>Replay “verified” only describes technical verification. A legally confirmed result requires a Rule Engine output and rule ID. Official citations are not exposed by the current contracts.</p>`,
  );
}

const actions = {
  theme: () => {
    state.theme = state.theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("banking-theme", state.theme);
    } catch {}
    render();
  },
  "dismiss-error": () => {
    state.error = null;
    render();
  },
  "zoom-image": () => {
    const original = $("#dialog").classList.toggle("actual-size");
    $('[data-action="zoom-image"]').innerHTML =
      icon("zoom-in") + (original ? "Fit" : "100%");
  },
  menu: () => {
    state.mobileNav = !state.mobileNav;
    render();
    $(state.mobileNav ? ".sidebar .active" : ".mobile-menu")?.focus();
  },
  help,
  upload: uploadDialog,
  "choose-video": () => $("#video-input").click(),
  "import-report": () => {
    if (!state.uploading) $("#report-input").click();
  },
  "choose-frames": () => {
    if (state.mode !== "demo") $("#frames-input").click();
  },
  "close-dialog": () => {
    if (!state.uploading) $("#dialog").close();
  },
  "connect-settings": () => {
    $("#dialog").close();
    navigate("settings");
  },
  "start-analysis": startAnalysis,
  demo: () => {
    if (state.uploading) return;
    resetSource("demo", demoAnalyses());
    navigate("analysis");
    toast("Demo workspace loaded. All values are synthetic.");
  },
  refresh: () => refresh(true),
  "open-analysis": () => navigate("analysis"),
  "open-evidence": () => navigate("evidence"),
  "open-legal": () => navigate("legal"),
  "open-events": () => navigate("events"),
  play: () => {
    const v = $("#source-video");
    if (v) {
      if (v.paused)
        v.play().catch(() =>
          toast("Playback is unavailable for this video codec."),
        );
      else v.pause();
    }
  },
  back: () => seekTo(Math.max(0, state.time - 5)),
  forward: () =>
    seekTo(
      Math.min(context().timelineDuration || state.time + 5, state.time + 5),
    ),
  fullscreen: async () => {
    try {
      await $("#video-stage")?.requestFullscreen();
    } catch {
      toast("Full screen is unavailable in this browser context.");
    }
  },
  "prev-event": () => cycleEvent(-1),
  "next-event": () => cycleEvent(1),
};

function cycleEvent(step) {
  const events = context().current?.events || [];
  if (!events.length) return;
  const index = events.findIndex((e) => e.event_id === state.selectedEvent);
  selectEvent(events[(index + step + events.length) % events.length].event_id);
}

document.addEventListener("click", (event) => {
  const target = event.target.closest(
    "[data-action],[data-page],[data-event],[data-filter],[data-analysis],[data-role],[data-legacy]",
  );
  if (!target || target.disabled) return;
  if (target.dataset.page) {
    event.preventDefault();
    navigate(target.dataset.page);
    return;
  }
  if (target.dataset.event) {
    selectEvent(target.dataset.event);
    return;
  }
  if (target.dataset.analysis) {
    selectAnalysis(target.dataset.analysis, true);
    if (state.mode === "api") refresh();
    return;
  }
  if (target.dataset.filter) {
    state.filter = target.dataset.filter;
    render();
    return;
  }
  if (target.dataset.role) {
    showFrame(target.dataset.role);
    return;
  }
  if (target.dataset.legacy != null) {
    const frame =
      context().event?.legacyFrames?.[Number(target.dataset.legacy)];
    const url = frame ? context().frameURL(frame) : null;
    if (url)
      modal(
        `Source frame · ${timestamp(frame.timestamp_sec, true)}`,
        `<img src="${h(url)}" alt="Original evidence frame"><div class="zoom-caption">Temporal role was not supplied. Verify this frame against the source recording.</div>`,
        "zoom-dialog",
      );
    else toast("Attach the original source image to inspect this frame.");
    return;
  }
  if (target.dataset.action === "review-event") {
    selectEvent(target.dataset.id, true);
    return;
  }
  actions[target.dataset.action]?.();
});

document.addEventListener("change", (event) => {
  if (event.target.id === "analysis-select") {
    selectAnalysis(event.target.value);
    if (state.mode === "api") refresh();
  }
  if (event.target.id === "event-select") selectEvent(event.target.value);
});
document.addEventListener("input", (event) => {
  if (event.target.id === "seek") seekTo(Number(event.target.value));
  if (event.target.id === "event-search") {
    const pos = event.target.selectionStart;
    state.query = event.target.value;
    render();
    const input = $("#event-search");
    input.focus({ preventScroll: true });
    input.setSelectionRange(pos, pos);
  }
});
document.addEventListener("submit", (event) => {
  if (event.target.id === "connection-form") {
    event.preventDefault();
    connect(event.target);
  }
});
document.addEventListener(
  "error",
  (event) => {
    const image = event.target;
    if (!(image instanceof HTMLImageElement)) return;
    state.imageFailures.add(image.src);
    image.replaceWith(
      document.createTextNode("Source image could not be decoded."),
    );
    toast(
      "A source image is invalid or unsupported. Attach the original JPEG or PNG.",
    );
  },
  true,
);
$("#video-input").addEventListener("change", (event) => {
  attachVideo(event.target.files[0]);
  event.target.value = "";
});
$("#report-input").addEventListener("change", (event) => {
  importReport(event.target.files[0]).catch(() =>
    toast("The report could not be read."),
  );
  event.target.value = "";
});
$("#frames-input").addEventListener("change", (event) => {
  const incoming = [...event.target.files].filter(
    (f) =>
      /\.(jpe?g|png)$/i.test(f.name) &&
      f.size > 0 &&
      f.size <= 24 * 1024 * 1024,
  );
  if (incoming.length !== event.target.files.length)
    toast("Skipped invalid images. Use JPEG or PNG files up to 24 MiB each.");
  state.fileURLs.forEach((url) => URL.revokeObjectURL(url));
  state.fileURLs.clear();
  state.files = incoming;
  incoming.forEach((f) => state.fileURLs.set(f, URL.createObjectURL(f)));
  event.target.value = "";
  render();
  toast(
    `${incoming.length} local source images attached. Verify their identity before drawing conclusions.`,
  );
});
$("#dialog").addEventListener("cancel", (event) => {
  if (state.uploading) event.preventDefault();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && state.mobileNav) {
    actions.menu();
  }
  if (event.key === "Tab" && state.mobileNav) {
    const items = [...document.querySelectorAll(".sidebar a,.sidebar button")];
    const first = items[0],
      last = items.at(-1);
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    }
    if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
});
document.addEventListener("dragover", (event) => {
  if ([...event.dataTransfer.types].includes("Files")) {
    event.preventDefault();
    $("#upload-zone")?.classList.add("drag-over");
  }
});
document.addEventListener("dragleave", () =>
  $("#upload-zone")?.classList.remove("drag-over"),
);
document.addEventListener("drop", (event) => {
  event.preventDefault();
  $("#upload-zone")?.classList.remove("drag-over");
  attachVideo(event.dataTransfer.files[0]);
});
window.addEventListener("hashchange", () => {
  const page = location.hash.slice(1);
  if (views[page] && state.page !== page) navigate(page);
});
window.addEventListener("message", (event) => {
  if (event.source !== window.parent || event.data?.type !== "streamlit:render")
    return;
  const args = event.data.args || {};
  state.bridgeReady = true;
  if (Number.isFinite(args.budget_seconds) && args.budget_seconds > 0)
    state.budget = args.budget_seconds;
  const response = args.report_response;
  // Streamlit may remount an iframe during a rerun; retain the validated server
  // response even when the browser's in-memory request ID was reset by that remount.
  if (shouldAcceptReport(response, state.requestId, state.lastResponseId)) {
    state.lastResponseId = response.id;
    state.requestId = null;
    state.loading = false;
    if (response.error) {
      state.error = response.error;
      render();
      return;
    }
    resetSource("report", [
      normalizeReport(
        response.report,
        state.importName || "Local pipeline report",
      ),
    ]);
    navigate("analysis");
    toast(
      "Local pipeline report validated. Attach the source recording and original frames to review.",
    );
  } else render();
});
window.addEventListener("beforeunload", () => {
  stopPolling();
  clearAttachments();
});
window.addEventListener("resize", () => {
  if (window.innerWidth >= 768 && state.mobileNav) {
    state.mobileNav = false;
    render();
  }
  updateHeight();
});
render();
send("streamlit:componentReady", { apiVersion: 1 });
updateHeight();
