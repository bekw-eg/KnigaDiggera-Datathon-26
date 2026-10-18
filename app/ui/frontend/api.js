/** Adapter for the routes in app/api/routes.py. No fallback endpoints. */
export class ApiError extends Error {
  constructor(message, code = "connection_error") {
    super(message);
    this.code = code;
  }
}

export function loopbackEndpoint(raw) {
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new ApiError("Enter a valid local API URL.");
  }
  if (
    url.protocol !== "http:" ||
    !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    url.pathname !== "/"
  ) {
    throw new ApiError(
      "Use a loopback HTTP origin, for example http://127.0.0.1:8000.",
    );
  }
  return url.origin;
}

const statuses = new Set(["queued", "processing", "done", "failed", "error"]);
function validJob(job) {
  if (
    !job ||
    typeof job.job_id !== "string" ||
    !job.job_id ||
    !statuses.has(job.status)
  ) {
    throw new ApiError(
      "The API returned an invalid job record.",
      "invalid_response",
    );
  }
  return job;
}

/** Public API responses lack CandidateEvent/VerifiedEvent, replay and legal citations. */
export function normalizeJob(raw, detail = false) {
  const job = validJob(raw);
  if (
    job.verdict != null &&
    !["violation", "no_violation"].includes(job.verdict)
  ) {
    throw new ApiError(
      "Invalid verdict enum in API response.",
      "invalid_response",
    );
  }
  if (
    detail &&
    job.status === "done" &&
    (job.verdict == null || !Array.isArray(job.violations))
  ) {
    throw new ApiError(
      "Completed job has no valid result. Check the backend result before retrying.",
      "invalid_response",
    );
  }
  if (job.violations != null && !Array.isArray(job.violations)) {
    throw new ApiError("Invalid events in API response.", "invalid_response");
  }
  const events = (job.violations || []).map((v, i) => {
    if (
      !v ||
      typeof v.type !== "string" ||
      !Number.isFinite(v.timecode_sec) ||
      v.timecode_sec < 0 ||
      typeof v.description !== "string" ||
      !Array.isArray(v.evidence_frames)
    ) {
      throw new ApiError(
        "Invalid event metadata in API response.",
        "invalid_response",
      );
    }
    const frames = v.evidence_frames.map((f) => {
      if (
        !f ||
        !Number.isFinite(f.timestamp_sec) ||
        f.timestamp_sec < 0 ||
        typeof f.image_base64 !== "string" ||
        f.image_base64.length > 24 * 1024 * 1024 ||
        !/^[A-Za-z0-9+/]*={0,2}$/.test(f.image_base64)
      ) {
        throw new ApiError(
          "Invalid evidence frame in API response.",
          "invalid_response",
        );
      }
      // Current API does not provide temporal roles. Never invent Before/During/After.
      return {
        timestamp_sec: f.timestamp_sec,
        role: null,
        image: f.image_base64
          ? `data:image/jpeg;base64,${f.image_base64}`
          : null,
      };
    });
    return {
      event_id: `${job.job_id}:item-${i + 1}`,
      event_type: v.type,
      start_sec: v.timecode_sec,
      end_sec: v.timecode_sec,
      roi_id: null,
      observations: {},
      verification: null,
      replay: null,
      legacyFrames: frames,
      apiDescription: v.description,
      apiSignal: true,
      frame_paths: [],
    };
  });
  return {
    id: job.job_id,
    name:
      job.video_path?.replaceAll("\\", "/").split("/").at(-1) ||
      `Job ${job.job_id.slice(0, 8)}`,
    source: "api",
    status: { done: "complete", error: "failed" }[job.status] || job.status,
    events,
    duration: null,
    metrics: null,
    coverage: null,
    stages: null,
    created: job.created_at,
    updated: job.updated_at,
    diagnostics: job.error_message ? [{ code: "backend_job_error" }] : [],
  };
}

export class LocalApiAdapter {
  constructor(endpoint, key) {
    this.endpoint = loopbackEndpoint(endpoint);
    this.key = key;
  }

  async request(path, authenticated = true) {
    let response;
    try {
      response = await fetch(`${this.endpoint}${path}`, {
        headers: authenticated ? { "X-API-Key": this.key } : {},
        signal: AbortSignal.timeout(10000),
        redirect: "error",
        cache: "no-store",
        credentials: "omit",
      });
    } catch {
      throw new ApiError(
        "Local API is unreachable or the request timed out. Check the service and retry.",
      );
    }
    if (!response.ok)
      throw new ApiError(
        response.status === 401
          ? "API key rejected. Check X-API-Key in Settings."
          : `Local API returned HTTP ${response.status}.`,
        `http_${response.status}`,
      );
    try {
      return await response.json();
    } catch {
      throw new ApiError("API response is not valid JSON.", "invalid_response");
    }
  }

  async health() {
    const health = await this.request("/health", false);
    if (
      health?.status !== "ok" ||
      !Number.isInteger(health.queue_size) ||
      health.queue_size < 0
    ) {
      throw new ApiError("Invalid health response.", "invalid_response");
    }
    return health;
  }

  async list() {
    const jobs = await this.request("/api/v1/jobs");
    if (!Array.isArray(jobs))
      throw new ApiError("Invalid job list.", "invalid_response");
    return jobs.map((job) => normalizeJob(job));
  }

  async get(id) {
    return normalizeJob(
      await this.request(`/api/v1/jobs/${encodeURIComponent(id)}`),
      true,
    );
  }

  /** Real byte-transfer progress only. Server analysis has no stage progress endpoint. */
  upload(file, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${this.endpoint}/api/v1/jobs/upload`);
      xhr.setRequestHeader("X-API-Key", this.key);
      xhr.timeout = 60000;
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total);
      };
      xhr.onerror = () =>
        reject(
          new ApiError(
            "Upload connection failed. Check the job list before retrying.",
          ),
        );
      xhr.ontimeout = () =>
        reject(
          new ApiError(
            "Upload timed out. The server may have accepted the file; refresh jobs before retrying.",
          ),
        );
      xhr.onload = () => {
        if (xhr.status !== 202) {
          reject(new ApiError(`Upload returned HTTP ${xhr.status}.`));
          return;
        }
        try {
          resolve(normalizeJob(JSON.parse(xhr.responseText)));
        } catch {
          reject(
            new ApiError(
              "Upload response is invalid. Refresh jobs before retrying.",
              "invalid_response",
            ),
          );
        }
      };
      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    });
  }
}
