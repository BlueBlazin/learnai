import { describe, expect, it } from "vitest";
import type { Path, ProblemStatus, ProblemSummary, Submission, Track } from "@/lib/api";
import { pickContinue } from "./continue";
import { findOnPath, nextOnPath, pathEntries, romanNumeral, unmetOffPath, unmetPrerequisites } from "./paths";

const problem = (
  id: string,
  status: ProblemStatus = "todo",
  over: Partial<ProblemSummary> = {},
): ProblemSummary => ({
  id,
  title: id.toUpperCase(),
  track: "t",
  order: 10,
  difficulty: "easy",
  frameworks: ["torch"],
  judges: ["numeric"],
  tags: [],
  prerequisites: [],
  requires: "cpu",
  status,
  solvedFrameworks: status === "solved" ? ["torch"] : [],
  assets: [],
  ...over,
});

const TRACKS: Track[] = [{ id: "t", title: "T", description: "", order: 1, problemCount: 5, solvedCount: 0 }];

const PATH: Path = {
  id: "p",
  title: "P",
  description: "",
  stages: [
    { title: "One", description: "", problems: ["c", "a"] },
    { title: "Two", description: "", problems: ["gone", "d", "b"] },
  ],
  problemCount: 4,
  solvedCount: 0,
};

const submission = (problemId: string, id = 1): Submission => ({
  id,
  problemId,
  framework: "torch",
  mode: "submit",
  verdict: "wrong_answer",
  passed: 0,
  total: 1,
  skipped: 0,
  durationMs: 1,
  target: "local",
  createdAt: "2026-11-14T10:00:00.000Z",
  review: false,
});

describe("pathEntries", () => {
  it("numbers entries across stages in path order and skips ids the catalog no longer has", () => {
    const entries = pathEntries(PATH, [problem("a"), problem("b"), problem("c"), problem("d")]);
    expect(entries.map((stage) => stage.map((e) => [e.problem.id, e.position, e.stage]))).toEqual([
      [
        ["c", 1, 0],
        ["a", 2, 0],
      ],
      [
        ["d", 3, 1],
        ["b", 4, 1],
      ],
    ]);
    expect(findOnPath(entries, "d")?.position).toBe(3);
    expect(findOnPath(entries, "zzz")).toBeNull();
  });

  it("finds the first unsolved entry, counting an attempted problem as unsolved", () => {
    const at = (statuses: Record<string, ProblemStatus>) =>
      nextOnPath(
        pathEntries(
          PATH,
          ["a", "b", "c", "d"].map((id) => problem(id, statuses[id] ?? "todo")),
        ),
      )?.problem.id ?? null;
    expect(at({})).toBe("c");
    expect(at({ c: "solved" })).toBe("a");
    expect(at({ c: "solved", a: "attempted" })).toBe("a");
    expect(at({ c: "solved", a: "solved", b: "solved" })).toBe("d");
    expect(at({ a: "solved", b: "solved", c: "solved", d: "solved" })).toBeNull();
  });
});

describe("pickContinue with a path", () => {
  const all = (statuses: Record<string, ProblemStatus> = {}) =>
    ["a", "b", "c", "d", "e"].map((id) => problem(id, statuses[id] ?? "todo"));

  it("starts a new user at the first problem on the path, not the first in the catalog", () => {
    const target = pickContinue(all(), TRACKS, [], PATH);
    expect(target?.problem.id).toBe("c");
    expect(target?.onPath).toEqual({ path: PATH, position: 1 });
    expect(pickContinue(all(), TRACKS, [], null)?.problem.id).toBe("a");
  });

  it("moves to the next unsolved problem on the path after a solve", () => {
    const target = pickContinue(all({ c: "solved" }), TRACKS, [], PATH);
    expect(target?.problem.id).toBe("a");
    expect(target?.onPath?.position).toBe(2);
    expect(target?.lastSubmission).toBeUndefined();
  });

  it("prefers the path over an attempted problem that is off it", () => {
    expect(pickContinue(all({ e: "attempted" }), TRACKS, [], PATH)?.problem.id).toBe("c");
  });

  it("still resumes unfinished submitted work first, and says where it sits on the path", () => {
    const onPath = pickContinue(all({ d: "attempted" }), TRACKS, [submission("d")], PATH);
    expect(onPath?.problem.id).toBe("d");
    expect(onPath?.lastSubmission?.problemId).toBe("d");
    expect(onPath?.onPath?.position).toBe(3);
    const offPath = pickContinue(all({ e: "attempted" }), TRACKS, [submission("e")], PATH);
    expect(offPath?.problem.id).toBe("e");
    expect(offPath?.onPath).toBeUndefined();
  });

  it("falls back to the catalog once the path is finished", () => {
    const done = all({ a: "solved", b: "solved", c: "solved", d: "solved" });
    const target = pickContinue(done, TRACKS, [], PATH);
    expect(target?.problem.id).toBe("e");
    expect(target?.onPath).toBeUndefined();
    expect(
      pickContinue(
        all({ a: "solved", b: "solved", c: "solved", d: "solved", e: "solved" }),
        TRACKS,
        [],
        PATH,
      ),
    ).toBeNull();
  });
});

describe("unmetPrerequisites", () => {
  it("lists unsolved prerequisites of unsolved problems only", () => {
    const problems = [
      problem("a", "solved"),
      problem("b", "attempted"),
      problem("c", "todo", { prerequisites: ["a"] }),
      problem("d", "todo", { prerequisites: ["a", "b", "missing"] }),
      problem("e", "solved", { prerequisites: ["b"] }),
      problem("f", "attempted", { prerequisites: ["c", "b"] }),
    ];
    const unmet = unmetPrerequisites(problems);
    expect([...unmet.keys()]).toEqual(["d", "f"]);
    expect(unmet.get("d")?.map((p) => p.id)).toEqual(["b"]);
    expect(unmet.get("f")?.map((p) => p.id)).toEqual(["c", "b"]);
  });
});

describe("unmetOffPath", () => {
  it("keeps only prerequisites the path does not already put first", () => {
    const problems = [
      problem("a"),
      problem("b", "todo", { prerequisites: ["d"] }),
      problem("c"),
      problem("d", "todo", { prerequisites: ["c", "e"] }),
      problem("e"),
    ];
    // Path order: c, a, d, b. `e` is off the path; `d` comes before `b`; `c` before `d`.
    const remaining = unmetOffPath(unmetPrerequisites(problems), pathEntries(PATH, problems));
    expect([...remaining.keys()]).toEqual(["d"]);
    expect(remaining.get("d")?.map((p) => p.id)).toEqual(["e"]);
  });
});

describe("romanNumeral", () => {
  it("numbers parts", () => {
    expect([1, 2, 3, 4, 5, 9, 10, 14, 40].map(romanNumeral)).toEqual([
      "I",
      "II",
      "III",
      "IV",
      "V",
      "IX",
      "X",
      "XIV",
      "XL",
    ]);
  });
});
