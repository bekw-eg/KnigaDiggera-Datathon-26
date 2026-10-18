import test from "node:test";
import assert from "node:assert/strict";
import {
  timestamp,
  legalState,
  filteredEvents,
  matchEvidence,
  demoAnalyses,
  summarize,
  shouldAcceptReport,
  escapeHTML,
} from "../app/ui/frontend/data.js";
import {
  loopbackEndpoint,
  normalizeJob,
  LocalApiAdapter,
} from "../app/ui/frontend/api.js";

test("presentation preserves fractional source timestamps and unknown values", () => {
  assert.equal(timestamp(68.25, true), "01:08.2");
  assert.equal(timestamp(3661.5, true), "01:01:01.5");
  for (const value of [NaN, Infinity, -1, null])
    assert.equal(timestamp(value), "—");
  assert.equal(filteredEvents(demoAnalyses()[0].events, "unknown").length, 4);
});
test("technical replay verified never becomes legal confirmed", () => {
  assert.equal(legalState({ replay: { status: "verified" } }), "uncertain");
  assert.equal(
    legalState({ verification: { state: "confirmed" } }),
    "uncertain",
  );
  assert.equal(
    legalState({ verification: { state: "confirmed", rule_id: "SYNTHETIC" } }),
    "confirmed",
  );
});
test("demo metrics derive from isolated records with no evidence pixels", () => {
  const records = demoAnalyses(),
    summary = summarize(records);
  assert.equal(summary.total, 4);
  assert.equal(summary.events, 5);
  assert.equal(summary.confirmed, 0);
  assert.equal(summary.uncertain, 4);
  assert.equal(summary.rejected, 1);
  assert.ok(
    records.flatMap((r) => r.events).every((e) => e.frame_paths.length === 0),
  );
});
test("ambiguous image basenames are not attached as evidence", () => {
  const files = [{ name: "frame.jpg" }, { name: "frame.jpg" }];
  assert.equal(matchEvidence("private/path/frame.jpg", files), null);
  assert.equal(matchEvidence("private/path/other.jpg", files), null);
  assert.equal(matchEvidence("private\\path\\frame.jpg", [files[0]]), files[0]);
});
test("all remotely configurable endpoints are rejected before a request", () => {
  assert.equal(
    loopbackEndpoint("http://127.0.0.1:8000"),
    "http://127.0.0.1:8000",
  );
  assert.equal(loopbackEndpoint("http://[::1]:8000"), "http://[::1]:8000");
  for (const url of [
    "https://localhost:8000",
    "http://example.com",
    "http://127.0.0.1.example.com",
    "http://user:secret@localhost:8000",
    "http://localhost:8000/other",
    "http://localhost:8000/?key=secret",
    "http://localhost:8000/#x",
  ]) {
    assert.throws(() => loopbackEndpoint(url));
  }
});
test("legacy API signal retains timestamp and never fabricates rule or temporal role", () => {
  const job = normalizeJob({
    job_id: "test",
    status: "done",
    violations: [
      {
        type: "test_signal",
        timecode_sec: 12.25,
        description: "Test only",
        evidence_frames: [{ timestamp_sec: 12.25, image_base64: "" }],
      },
    ],
  });
  const event = job.events[0];
  assert.equal(event.start_sec, 12.25);
  assert.equal(event.legacyFrames[0].role, null);
  assert.equal(legalState(event), "uncertain");
  assert.equal(job.coverage, null);
  assert.equal(job.stages, null);
});
test("invalid timestamps / enums / evidence are errors, never successful empty results", () => {
  assert.throws(() => normalizeJob({ job_id: "test", status: "invented" }));
  for (const t of [-1, NaN, Infinity, "12"])
    assert.throws(() =>
      normalizeJob({
        job_id: "x",
        status: "done",
        violations: [
          { type: "x", timecode_sec: t, description: "", evidence_frames: [] },
        ],
      }),
    );
  assert.throws(() =>
    normalizeJob({
      job_id: "x",
      status: "done",
      violations: [
        {
          type: "x",
          timecode_sec: 0,
          description: "",
          evidence_frames: [{ timestamp_sec: 0, image_base64: "<script>" }],
        },
      ],
    }),
  );
});
test("adapter uses exact existing paths and handles auth and malformed responses", async () => {
  const saved = globalThis.fetch;
  let called;
  try {
    globalThis.fetch = async (url, options) => {
      called = { url, options };
      return {
        ok: true,
        json: async () => ({ job_id: "a/b", status: "queued" }),
      };
    };
    const adapter = new LocalApiAdapter("http://localhost:8000", "test-key");
    await adapter.get("a/b");
    assert.equal(called.url, "http://localhost:8000/api/v1/jobs/a%2Fb");
    assert.equal(called.options.headers["X-API-Key"], "test-key");
    assert.equal(called.options.redirect, "error");
    globalThis.fetch = async () => ({ ok: false, status: 401 });
    await assert.rejects(adapter.list(), /API key rejected/);
    globalThis.fetch = async () => {
      throw new Error("network");
    };
    await assert.rejects(adapter.list(), /unreachable/);
    globalThis.fetch = async () => ({
      ok: true,
      json: async () => {
        throw new SyntaxError();
      },
    });
    await assert.rejects(adapter.list(), /valid JSON/);
  } finally {
    globalThis.fetch = saved;
  }
});
test("user-provided labels are escaped as text", () => {
  assert.equal(
    escapeHTML('<img onerror="x">'),
    "&lt;img onerror=&quot;x&quot;&gt;",
  );
});
test("report validation responses survive iframe remounts and are not replayed", () => {
  assert.equal(shouldAcceptReport({ id: "a" }, "a", null), true);
  assert.equal(shouldAcceptReport({ id: "a" }, null, null), true);
  assert.equal(shouldAcceptReport({ id: "a" }, null, "a"), false);
  assert.equal(shouldAcceptReport({ id: "a" }, "b", null), false);
  assert.equal(shouldAcceptReport(null, null, null), false);
});
test("completed API detail without a valid result is an error; summaries remain valid", () => {
  assert.equal(
    normalizeJob({ job_id: "x", status: "done" }).status,
    "complete",
  );
  assert.throws(
    () =>
      normalizeJob(
        { job_id: "x", status: "done", verdict: null, violations: [] },
        true,
      ),
    /no valid result/,
  );
  assert.throws(
    () =>
      normalizeJob(
        { job_id: "x", status: "done", verdict: "probably", violations: [] },
        true,
      ),
    /Invalid verdict/,
  );
  assert.equal(
    normalizeJob(
      { job_id: "x", status: "done", verdict: "no_violation", violations: [] },
      true,
    ).events.length,
    0,
  );
});
