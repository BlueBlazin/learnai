import { describe, expect, it } from "vitest";
import {
  initialTargetCheckState,
  stepProgress,
  type TargetCheckAction,
  type TargetCheckState,
  targetCheckReducer,
} from "./target-check-state";
import type { CheckStep, TargetCheck } from "./types";

const pass = (id: CheckStep["id"]): CheckStep => ({ id, ok: true, message: `${id} ok`, fix: null });
const fail = (id: CheckStep["id"]): CheckStep => ({ id, ok: false, message: `${id} failed`, fix: "do this" });

const fold = (actions: TargetCheckAction[], from: TargetCheckState = initialTargetCheckState) =>
  actions.reduce(targetCheckReducer, from);

describe("targetCheckReducer", () => {
  it("collects steps and log lines until done", () => {
    const check: TargetCheck = {
      ok: true,
      checkedAt: "2026-11-15T10:00:00.000Z",
      steps: [pass("connect"), pass("uv")],
      doctor: null,
    };
    const running = fold([
      { type: "begin" },
      { type: "event", event: { type: "step", step: pass("connect") } },
      { type: "event", event: { type: "log", text: "Resolved 61 packages" } },
    ]);
    expect(running).toMatchObject({ running: true, steps: [pass("connect")], log: ["Resolved 61 packages"] });

    const done = fold([{ type: "event", event: { type: "done", check } }], running);
    expect(done).toMatchObject({ running: false, steps: check.steps, result: check, error: null });
    // Nothing follows `done`.
    expect(fold([{ type: "event", event: { type: "step", step: pass("sync") } }], done)).toBe(done);
  });

  it("starts clean, ignores a second begin, and reports a check that could not start", () => {
    const failed = fold([{ type: "begin" }, { type: "failed", message: "unknown target" }]);
    expect(failed).toMatchObject({ running: false, error: "unknown target" });
    const again = fold([{ type: "begin" }], failed);
    expect(again).toEqual({ ...initialTargetCheckState, running: true });
    expect(fold([{ type: "begin" }], again)).toBe(again);
    expect(fold([{ type: "failed", message: "late" }])).toBe(initialTargetCheckState);
  });

  it("keeps a bounded log", () => {
    const lines = Array.from({ length: 450 }, (_, index) => ({
      type: "event" as const,
      event: { type: "log" as const, text: `line ${index}` },
    }));
    const state = fold([{ type: "begin" }, ...lines]);
    expect(state.log).toHaveLength(400);
    expect(state.log.at(-1)).toBe("line 449");
  });
});

describe("stepProgress", () => {
  const states = (kind: "local" | "ssh", steps: CheckStep[], running: boolean) =>
    stepProgress(kind, steps, running).map((entry) => `${entry.id}:${entry.state}`);

  it("shows the step in flight and the ones to come while running", () => {
    expect(states("ssh", [], true)).toEqual([
      "connect:running",
      "uv:pending",
      "sync:pending",
      "environment:pending",
      "doctor:pending",
    ]);
    expect(states("ssh", [pass("connect"), pass("uv")], true)).toEqual([
      "connect:done",
      "uv:done",
      "sync:running",
      "environment:pending",
      "doctor:pending",
    ]);
  });

  it("marks the steps after a failure as not reached", () => {
    expect(states("ssh", [pass("connect"), fail("uv")], false)).toEqual([
      "connect:done",
      "uv:done",
      "sync:skipped",
      "environment:skipped",
      "doctor:skipped",
    ]);
    // The failing step has arrived but `done` has not: nothing else is in flight.
    expect(states("ssh", [fail("connect")], true)).toEqual([
      "connect:done",
      "uv:skipped",
      "sync:skipped",
      "environment:skipped",
      "doctor:skipped",
    ]);
  });

  it("plans two steps for this computer", () => {
    expect(states("local", [pass("uv")], true)).toEqual(["uv:done", "doctor:running"]);
    expect(states("local", [pass("uv"), pass("doctor")], false)).toEqual(["uv:done", "doctor:done"]);
  });
});
