import { describe, expect, it } from "vitest";
import type { CaseStatus } from "@/lib/api";
import { groupByStage, type ListedCase, stageComplete, stageCount, stageHeading } from "./stages";

const listed = (
  id: string,
  stage: string | undefined,
  status: CaseStatus | "running" | "pending",
): ListedCase => ({
  info: { id, name: id, public: true, ...(stage ? { stage } : {}) },
  state: status === "running" || status === "pending" ? status : "done",
  result:
    status === "running" || status === "pending"
      ? null
      : { id, status, durationMs: 1, message: null, detail: null, stdout: "", metrics: [] },
});

describe("stage headings", () => {
  it("prefixes a bare label and leaves one that already says Stage", () => {
    expect(stageHeading("1: Tokenizer")).toBe("Stage 1: Tokenizer");
    expect(stageHeading("A")).toBe("Stage A");
    expect(stageHeading("Stage 2: Sampler")).toBe("Stage 2: Sampler");
    expect(stageHeading("stage three")).toBe("stage three");
    // "Staged rollout" does not begin with the word "Stage".
    expect(stageHeading("Staged rollout")).toBe("Stage Staged rollout");
  });
});

describe("groupByStage", () => {
  it("leaves an unstaged problem alone", () => {
    expect(groupByStage([listed("c1", undefined, "pass"), listed("c2", undefined, "fail")])).toBeNull();
    expect(groupByStage([])).toBeNull();
  });

  it("groups in first-appearance order, even when a stage's cases are not adjacent", () => {
    const groups = groupByStage([
      listed("a1", "2: Sampler", "pass"),
      listed("b1", "1: Tokenizer", "pass"),
      listed("a2", "2: Sampler", "fail"),
      listed("b2", "1: Tokenizer", "pass"),
    ]);
    expect(groups?.map((g) => [g.heading, g.cases.map((c) => c.info.id)])).toEqual([
      ["Stage 2: Sampler", ["a1", "a2"]],
      ["Stage 1: Tokenizer", ["b1", "b2"]],
    ]);
  });

  it("counts passed and skipped per stage, live and ended", () => {
    const groups = groupByStage([
      listed("t1", "1: Tokenizer", "pass"),
      listed("t2", "1: Tokenizer", "pass"),
      listed("d1", "Stage 2: Decode", "pass"),
      listed("d2", "Stage 2: Decode", "running"),
      listed("d3", "Stage 2: Decode", "pending"),
      listed("s1", "3: Serving", "error"),
      listed("s2", "3: Serving", "skipped"),
      listed("s3", "3: Serving", "timeout"),
    ]);
    expect(groups?.map((g) => [g.label, g.passed, g.skipped, g.total])).toEqual([
      ["1: Tokenizer", 2, 0, 2],
      ["Stage 2: Decode", 1, 0, 3],
      ["3: Serving", 0, 1, 3],
    ]);
    expect(groups?.map(stageCount)).toEqual(["2/2 passed", "1/3 passed", "0/3 passed, 1 skipped"]);
    expect(groups?.map(stageComplete)).toEqual([true, false, false]);
  });

  it("puts a case without a stage after the stages of a staged problem", () => {
    const groups = groupByStage([
      listed("x", undefined, "pass"),
      listed("t1", "1: Tokenizer", "pass"),
      listed("y", undefined, "fail"),
    ]);
    expect(groups?.map((g) => [g.heading, g.cases.map((c) => c.info.id)])).toEqual([
      ["Stage 1: Tokenizer", ["t1"]],
      ["Other cases", ["x", "y"]],
    ]);
    expect(groups?.[1] && stageCount(groups[1])).toBe("1/2 passed");
  });
});
