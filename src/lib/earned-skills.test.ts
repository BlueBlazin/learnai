import { describe, expect, it } from "vitest";
import type { Skill } from "@/lib/api";
import { initialRunState, type RunState } from "@/lib/api/run-state";
import { earnedReducer, earningSubmission } from "./earned-skills";

const skill = (id: string): Skill => ({
  id,
  title: id,
  name: id,
  summary: "",
  requires: [],
  problems: ["p"],
  row: 0,
  x: 500,
  lanes: [],
  earnedAt: "2026-11-15T10:00:00.000Z",
});

const ACCEPTED: RunState = {
  ...initialRunState,
  phase: "done",
  mode: "submit",
  verdict: "accepted",
  submissionId: 7,
};

describe("earningSubmission", () => {
  it("is the stored submission of an accepted submit that has ended", () => {
    expect(earningSubmission(ACCEPTED)).toBe(7);
  });

  it("is null before any run and while one is going", () => {
    expect(earningSubmission(initialRunState)).toBeNull();
    expect(
      earningSubmission({ ...ACCEPTED, phase: "running", verdict: null, submissionId: null }),
    ).toBeNull();
  });

  it("is null for a Run, a failed submit, and a review submit", () => {
    expect(earningSubmission({ ...ACCEPTED, mode: "run", submissionId: null })).toBeNull();
    expect(earningSubmission({ ...ACCEPTED, verdict: "wrong_answer" })).toBeNull();
    expect(earningSubmission({ ...ACCEPTED, review: true })).toBeNull();
  });

  it("is null for an accepted submit that was not stored", () => {
    expect(earningSubmission({ ...ACCEPTED, submissionId: null })).toBeNull();
  });
});

describe("earnedReducer", () => {
  const [a, b, c] = [skill("a"), skill("b"), skill("c")];

  it("shows the skills in the order they were returned, one per close", () => {
    let queue = earnedReducer([], { type: "earned", skills: [a, b] });
    expect(queue.map((s) => s.id)).toEqual(["a", "b"]);
    queue = earnedReducer(queue, { type: "next" });
    expect(queue.map((s) => s.id)).toEqual(["b"]);
    queue = earnedReducer(queue, { type: "next" });
    expect(queue).toEqual([]);
    expect(earnedReducer(queue, { type: "next" })).toEqual([]);
  });

  it("keeps the same queue when a submit earned nothing", () => {
    const queue = [a];
    expect(earnedReducer(queue, { type: "earned", skills: [] })).toBe(queue);
  });

  it("puts later skills behind the one showing and never lists one twice", () => {
    const queue = earnedReducer([a, b], { type: "earned", skills: [b, c] });
    expect(queue.map((s) => s.id)).toEqual(["a", "b", "c"]);
  });

  it("drops the rest on clear", () => {
    expect(earnedReducer([a, b], { type: "clear" })).toEqual([]);
    const empty: Skill[] = [];
    expect(earnedReducer(empty, { type: "clear" })).toBe(empty);
  });
});
