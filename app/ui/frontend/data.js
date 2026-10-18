/** Pure presentation transforms. No model inference or legal rules live here. */
export const STAGES = [
  "Video decoding",
  "ROI calibration validation",
  "Motion detection",
  "Temporal analysis",
  "Local VLM verification",
  "Evidence generation",
  "Legal rule evaluation",
  "Analysis completed",
];

export const TITLES = {
  motion_candidate: "Motion in observation zone",
  object_transfer: "Possible object transfer",
  person_presence: "Figure near glass",
  unknown_object: "Uncertain object observation",
  desk_activity: "Activity at desk",
};

export function title(type) {
  return (
    TITLES[type] ||
    String(type)
      .replaceAll("_", " ")
      .replace(/^./, (c) => c.toUpperCase())
  );
}

export function timestamp(seconds, decimal = false) {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor(seconds / 60) % 60;
  const sec = Math.floor(seconds % 60);
  return `${hours ? `${String(hours).padStart(2, "0")}:` : ""}${String(minutes).padStart(2, "0")}:${String(sec).padStart(2, "0")}${decimal ? `.${Math.floor((seconds % 1) * 10)}` : ""}`;
}

export function escapeHTML(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
}

/** Technical replay verified is deliberately never used to derive legal state. */
export function legalState(event) {
  const result = event?.verification;
  if (result?.state === "confirmed")
    return result.rule_id ? "confirmed" : "uncertain";
  return ["uncertain", "rejected"].includes(result?.state)
    ? result.state
    : "uncertain";
}

export function filteredEvents(events, filter, query = "") {
  const text = query.toLowerCase();
  return events.filter(
    (e) =>
      (filter === "all" ||
        (filter === "unknown"
          ? Object.values(e.observations).some((v) => v === "unknown")
          : legalState(e) === filter)) &&
      `${e.event_id} ${e.event_type} ${e.roi_id || ""}`
        .toLowerCase()
        .includes(text),
  );
}

export function normalizeReport(bundle, name = "Local pipeline report") {
  const result = bundle.analysis;
  return {
    id: `report-${Date.now()}`,
    name,
    status: result.status,
    source: "report",
    duration: result.metadata?.duration_sec ?? null,
    metrics: result.metrics,
    coverage: result.coverage,
    diagnostics: result.diagnostics,
    events: result.events.map((e) => ({
      ...e,
      legacyFrames: result.evidence_frames.filter(
        (f) =>
          e.frame_paths.includes(f.path) &&
          !result.event_replay
            .find((r) => r.event_id === e.event_id)
            ?.frames.some((r) => r.path === f.path),
      ),
      replay:
        result.event_replay.find((r) => r.event_id === e.event_id) || null,
      verification:
        bundle.verification.find((v) => v.event_id === e.event_id) || null,
    })),
    created: new Date().toISOString(),
    stages: null,
  };
}

/** Isolated test metadata. There are no demo video pixels or evidence images. */
export function demoAnalyses() {
  const specs = [
    [
      34.5,
      37.5,
      "object_transfer",
      "desk",
      {
        object_transfer: "unknown",
        transferred_object: "unknown",
        hand_visible: "yes",
      },
    ],
    [
      68.25,
      71.25,
      "person_presence",
      "glass_boundary",
      { physical_presence: "unknown", possible_reflection: "yes" },
    ],
    [
      112.75,
      115.75,
      "desk_activity",
      "desk",
      { object_transfer: "no", hand_visible: "yes" },
    ],
    [
      164.5,
      167.5,
      "unknown_object",
      null,
      { object_visible: "unknown", object_transfer: "unknown" },
    ],
    [
      198.25,
      201.25,
      "motion_candidate",
      "desk",
      { physical_presence: "unknown", object_transfer: "unknown" },
    ],
  ];
  const events = specs.map(([start, end, type, roi, observations], i) => ({
    event_id: `DEMO-${String(i + 1).padStart(3, "0")}`,
    event_type: type,
    start_sec: start,
    end_sec: end,
    roi_id: roi,
    frame_paths: [],
    observations,
    verification:
      i === 2
        ? {
            state: "rejected",
            rule_id: null,
            explanation:
              "Test state: this observation was not confirmed during review.",
          }
        : null,
    replay: {
      status: i === 2 ? "verified" : "incomplete",
      frames: [],
      diagnostic_codes:
        i === 1 ? ["possible_glass_reflection"] : ["demo_media_not_attached"],
    },
  }));
  return [
    {
      id: "demo-01",
      name: "Synthetic desk activity.mov",
      source: "demo",
      status: "partial",
      duration: 252,
      created: "2026-10-18T09:24:00+05:00",
      events,
      metrics: {
        total_sec: 142.6,
        sampled_frames: 424,
        decoded_frames: 6360,
        vlm_calls: 4,
        vlm_successes: 3,
      },
      coverage: {
        decoded_fraction: 0.84,
        decoded_until_sec: 211.68,
        decoded_complete: false,
        sample_fps: 2,
        vlm_intervals: [
          [34.5, 37.5],
          [68.25, 71.25],
          [112.75, 115.75],
        ],
      },
      diagnostics: [
        { code: "demo_metadata_only" },
        { code: "vlm_call_budget_reached" },
      ],
      stages: [
        "partial",
        "unavailable",
        "complete",
        "complete",
        "partial",
        "unavailable",
        "unavailable",
        "partial",
      ],
    },
    {
      id: "demo-02",
      name: "Synthetic quiet scene.mov",
      source: "demo",
      status: "complete",
      duration: 120,
      created: "2026-10-18T09:10:00+05:00",
      events: [],
      metrics: { total_sec: 38.2, sampled_frames: 240 },
      coverage: {
        decoded_fraction: 1,
        decoded_complete: true,
        sample_fps: 2,
        vlm_intervals: [],
      },
      diagnostics: [],
      stages: null,
    },
    {
      id: "demo-03",
      name: "Synthetic boundary test.mov",
      source: "demo",
      status: "complete",
      duration: 90,
      created: "2026-10-17T17:30:00+05:00",
      events: [],
      metrics: { total_sec: 51.4 },
      coverage: null,
      diagnostics: [],
      stages: null,
    },
    {
      id: "demo-04",
      name: "Synthetic timeout test.mov",
      source: "demo",
      status: "timed_out",
      duration: 600,
      created: "2026-10-17T16:15:00+05:00",
      events: [],
      metrics: { total_sec: 300 },
      coverage: null,
      diagnostics: [{ code: "deadline_exceeded" }],
      stages: null,
    },
  ];
}

/** Build KPIs only from records actually loaded in this workspace. */
export function summarize(analyses) {
  const events = analyses.flatMap((a) => a.events);
  const times = analyses
    .map((a) => a.metrics?.total_sec)
    .filter(Number.isFinite);
  return {
    total: analyses.length,
    processed: analyses.filter((a) => a.status === "complete").length,
    events: events.length,
    confirmed: events.filter((e) => legalState(e) === "confirmed").length,
    uncertain: events.filter((e) => legalState(e) === "uncertain").length,
    rejected: events.filter((e) => legalState(e) === "rejected").length,
    average: times.length
      ? times.reduce((a, b) => a + b, 0) / times.length
      : null,
  };
}

/** Evidence matching must be unique; duplicate basenames never silently select a frame. */
export function matchEvidence(path, files) {
  const normalized = String(path).replaceAll("\\", "/");
  const exact = files.filter(
    (f) => normalized === (f.webkitRelativePath || f.name),
  );
  if (exact.length === 1) return exact[0];
  const matches = files.filter((f) => normalized.split("/").at(-1) === f.name);
  return matches.length === 1 ? matches[0] : null;
}

/** A keyed component may remount during Streamlit reruns. Apply each response once. */
export function shouldAcceptReport(response, pendingId, lastId) {
  return (
    !!response?.id &&
    response.id !== lastId &&
    (!pendingId || response.id === pendingId)
  );
}
