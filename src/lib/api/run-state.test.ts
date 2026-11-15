import { describe, expect, it } from "vitest";
import { initialRunState, isActive, logText, type RunAction, type RunState, runReducer } from "./run-state";
import type { CaseResult, RunEvent } from "./types";

const result = (id: string, status: CaseResult["status"] = "pass"): CaseResult => ({
  id,
  status,
  durationMs: 5,
  message: status === "pass" ? null : "boom",
  detail: null,
  stdout: "",
  metrics: [],
});

const cases = [
  { id: "c1", name: "first", public: true },
  { id: "c2", name: "second", public: true },
  { id: "c3", name: "Hidden case 1", public: false },
];

const START: Extract<RunEvent, { type: "start" }> = {
  type: "start",
  cases,
  env: {},
  requires: "cpu",
  requiresMet: true,
};

function play(actions: RunAction[], from: RunState = initialRunState): RunState {
  return actions.reduce(runReducer, from);
}

const event = (e: RunEvent): RunAction => ({ type: "event", event: e });
const begin: RunAction = { type: "begin", mode: "submit", framework: "torch" };

describe("runReducer", () => {
  it("walks starting, preparing, running, done", () => {
    let state = play([begin]);
    expect(state.phase).toBe("starting");
    expect(isActive(state)).toBe(true);

    state = play([event({ type: "preparing", message: "Syncing" })], state);
    expect(state.phase).toBe("preparing");
    expect(state.preparingMessage).toBe("Syncing");

    state = play([event({ ...START, env: { python: "3.15.2" } })], state);
    expect(state.phase).toBe("running");
    expect(state.preparingMessage).toBeNull();
    expect(state.total).toBe(3);
    expect(state.cases.map((c) => c.state)).toEqual(["pending", "pending", "pending"]);

    state = play([event({ type: "caseStart", id: "c1" })], state);
    expect(state.cases[0]?.state).toBe("running");

    state = play([event({ type: "caseEnd", result: result("c1") })], state);
    expect(state.cases[0]?.state).toBe("done");
    expect(state.passed).toBe(1);

    state = play(
      [
        event({ type: "caseStart", id: "c2" }),
        event({ type: "caseEnd", result: result("c2", "fail") }),
        event({ type: "caseEnd", result: result("c3", "skipped") }),
        event({
          type: "end",
          verdict: "wrong_answer",
          passed: 1,
          total: 3,
          durationMs: 10,
          metrics: [{ name: "model_tokens", value: 41, limit: 36 }],
          submissionId: 7,
          skipped: 0,
          message: null,
          detail: null,
        }),
      ],
      state,
    );
    expect(state.phase).toBe("done");
    expect(isActive(state)).toBe(false);
    expect(state.verdict).toBe("wrong_answer");
    expect(state.submissionId).toBe(7);
    expect(state.metrics).toHaveLength(1);
    expect(state.cases.map((c) => c.result?.status)).toEqual(["pass", "fail", "skipped"]);
  });

  it("carries the message and detail of an import error, with cases skipped without a start", () => {
    const state = play([
      begin,
      event(START),
      ...cases.map((c) => event({ type: "caseEnd", result: result(c.id, "skipped") })),
      event({
        type: "end",
        verdict: "import_error",
        passed: 0,
        total: 3,
        durationMs: 2,
        metrics: [],
        submissionId: 8,
        skipped: 0,
        message: "the solution failed to import: SyntaxError: '(' was never closed",
        detail: '  File "torch.py", line 1\n',
      }),
    ]);
    expect(state.verdict).toBe("import_error");
    expect(state.message).toContain("SyntaxError");
    expect(state.detail).toContain("line 1");
    expect(state.cases.map((c) => [c.state, c.result?.status])).toEqual([
      ["done", "skipped"],
      ["done", "skipped"],
      ["done", "skipped"],
    ]);
  });

  it("keeps case order from `start` regardless of completion order", () => {
    const state = play([
      begin,
      event(START),
      event({ type: "caseEnd", result: result("c3") }),
      event({ type: "caseEnd", result: result("c1") }),
    ]);
    expect(state.cases.map((c) => c.info.id)).toEqual(["c1", "c2", "c3"]);
    expect(state.passed).toBe(2);
  });

  it("tolerates results for cases that were never announced", () => {
    const state = play([begin, event({ type: "caseEnd", result: result("x9", "error") })]);
    expect(state.cases).toHaveLength(1);
    expect(state.cases[0]?.info.name).toBe("x9");
  });

  it("returns a running case to pending when the run is canceled", () => {
    const state = play([
      begin,
      event(START),
      event({ type: "caseStart", id: "c1" }),
      { type: "canceling" },
      event({
        type: "end",
        verdict: "canceled",
        passed: 0,
        total: 3,
        durationMs: 0,
        metrics: [],
        submissionId: null,
        skipped: 0,
        message: null,
        detail: null,
      }),
    ]);
    expect(state.verdict).toBe("canceled");
    expect(state.canceling).toBe(false);
    expect(state.cases[0]?.state).toBe("pending");
  });

  it("ignores events after `end` and while idle", () => {
    const end = event({
      type: "end",
      verdict: "accepted",
      passed: 3,
      total: 3,
      durationMs: 1,
      metrics: [],
      submissionId: null,
      skipped: 0,
      message: null,
      detail: null,
    });
    const done = play([begin, event(START), end]);
    expect(play([event({ type: "caseStart", id: "c1" })], done)).toBe(done);
    expect(play([event({ type: "preparing", message: "late" })])).toBe(initialRunState);
  });

  it("records a failure to start as a judge failure with the message", () => {
    const state = play([begin, { type: "failed", message: "uv not found" }]);
    expect(state.phase).toBe("done");
    expect(state.verdict).toBe("harness_error");
    expect(state.startError).toBe("uv not found");
  });

  it("keeps the run-level message and detail sent with `end`", () => {
    const state = play([
      begin,
      event({
        type: "end",
        verdict: "import_error",
        passed: 0,
        total: 0,
        durationMs: 0,
        metrics: [],
        submissionId: null,
        skipped: 0,
        message: "ModuleNotFoundError: einops",
        detail: "Traceback",
      }),
    ]);
    expect(state.message).toBe("ModuleNotFoundError: einops");
    expect(state.detail).toBe("Traceback");
  });

  it("collects diagnostics by stream", () => {
    const state = play([
      begin,
      event({ type: "log", stream: "stderr", text: "warn" }),
      event({ type: "log", stream: "stdout", text: "out" }),
      event({ type: "log", stream: "stderr", text: "again" }),
    ]);
    expect(logText(state.logs, "stderr")).toBe("warn\nagain");
    expect(logText(state.logs)).toBe("warn\nout\nagain");
  });

  it("begin clears the previous run", () => {
    const done = play([begin, event(START), event({ type: "caseEnd", result: result("c1") })]);
    const again = play([{ type: "begin", mode: "run", framework: "jax" }], done);
    expect(again.cases).toEqual([]);
    expect(again.mode).toBe("run");
    expect(again.framework).toBe("jax");
  });

  it("keeps skipped cases apart from failures and carries a not-judged ending", () => {
    const skipped: CaseResult = {
      ...result("c3", "skipped"),
      message: "skipped: needs an NVIDIA GPU (CUDA), which this machine does not have",
      skip: { kind: "hardware", requires: "cuda" },
    };
    let state = play([
      begin,
      event({ ...START, requires: "cuda", requiresMet: false }),
      event({ type: "caseStart", id: "c1" }),
      event({ type: "caseEnd", result: result("c1") }),
      event({ type: "caseEnd", result: skipped }),
    ]);
    expect([state.requires, state.requiresMet]).toEqual(["cuda", false]);
    expect([state.passed, state.skipped]).toEqual([1, 1]);
    expect(state.cases[2]?.result?.skip).toEqual({ kind: "hardware", requires: "cuda" });

    state = play(
      [
        event({
          type: "end",
          verdict: "hardware_missing",
          passed: 2,
          total: 3,
          skipped: 1,
          durationMs: 9,
          metrics: [],
          submissionId: 4,
          missingHardware: "cuda",
          message: "2 of 2 cases that could run passed",
          detail: null,
        }),
      ],
      state,
    );
    expect(state.phase).toBe("done");
    expect(state.verdict).toBe("hardware_missing");
    expect(state.skipped).toBe(1);
    expect(state.missingHardware).toBe("cuda");
    expect(state.missingAssets).toEqual([]);
    expect(state.message).toBe("2 of 2 cases that could run passed");

    const assets = play([
      begin,
      event(START),
      event({
        type: "end",
        verdict: "assets_missing",
        passed: 1,
        total: 3,
        skipped: 2,
        durationMs: 9,
        metrics: [],
        submissionId: null,
        missingAssets: [{ name: "granite-4.0-350m", repo: "ibm-granite/granite-4.0-350m", sizeMb: 712 }],
        message: "m",
        detail: null,
      }),
    ]);
    expect(assets.missingAssets.map((asset) => asset.name)).toEqual(["granite-4.0-350m"]);
    expect(assets.missingHardware).toBeNull();
    // A new run starts clean.
    expect(play([begin], assets).missingAssets).toEqual([]);
  });
});

describe("review runs and staged cases", () => {
  const END: RunEvent = {
    type: "end",
    verdict: "accepted",
    passed: 3,
    total: 3,
    durationMs: 12,
    metrics: [],
    submissionId: 4,
    skipped: 0,
    message: null,
    detail: null,
  };

  it("remembers that a run judges a review attempt until the next run begins", () => {
    const review = play([{ ...begin, review: true }, event(START), event(END)]);
    expect(review.review).toBe(true);
    expect(review.phase).toBe("done");
    expect(play([begin], review).review).toBe(false);
    expect(play([begin]).review).toBe(false);
  });

  it("keeps the stage of each announced case as results stream in", () => {
    const staged = [
      { id: "t1", name: "tokenize", public: true, stage: "1: Tokenizer" },
      { id: "s1", name: "Hidden case 1", public: false, stage: "Stage 2: Sampler" },
    ];
    const state = play([
      begin,
      event({ ...START, cases: staged }),
      event({ type: "caseStart", id: "t1" }),
      event({ type: "caseEnd", result: result("t1") }),
      event({ type: "caseEnd", result: result("s1", "skipped") }),
    ]);
    expect(state.cases.map((c) => [c.info.stage, c.result?.status])).toEqual([
      ["1: Tokenizer", "pass"],
      ["Stage 2: Sampler", "skipped"],
    ]);
  });
});
