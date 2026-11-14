import type { Path, ProblemSummary } from "@/lib/api";

/** A problem at its place on a path. */
export interface PathEntry {
  problem: ProblemSummary;
  /** 1-based position along the whole path, across stages. */
  position: number;
  /** 0-based index of the entry's stage. */
  stage: number;
}

/**
 * The path as numbered entries, stage by stage. An id with no matching problem
 * (the catalog changed between two fetches) is left out, and its number with it.
 */
export function pathEntries(path: Path, problems: ProblemSummary[]): PathEntry[][] {
  const byId = new Map(problems.map((problem) => [problem.id, problem]));
  let position = 0;
  return path.stages.map((stage, index) =>
    stage.problems.flatMap((id) => {
      const problem = byId.get(id);
      if (!problem) return [];
      position += 1;
      return [{ problem, position, stage: index }];
    }),
  );
}

/** The suggested next step: the first entry on the path that is not solved. Null when all are. */
export function nextOnPath(entries: PathEntry[][]): PathEntry | null {
  for (const stage of entries) {
    const entry = stage.find((candidate) => candidate.problem.status !== "solved");
    if (entry) return entry;
  }
  return null;
}

/** Where `problemId` sits on the path, if it is on it. */
export function findOnPath(entries: PathEntry[][], problemId: string): PathEntry | null {
  for (const stage of entries) {
    const entry = stage.find((candidate) => candidate.problem.id === problemId);
    if (entry) return entry;
  }
  return null;
}

/**
 * For each problem with prerequisites that are not solved yet: those
 * prerequisites, in the order the problem lists them. Solved problems are left
 * out (there is nothing left to recommend), as are unknown ids.
 */
export function unmetPrerequisites(problems: ProblemSummary[]): Map<string, ProblemSummary[]> {
  const byId = new Map(problems.map((problem) => [problem.id, problem]));
  const unmet = new Map<string, ProblemSummary[]>();
  for (const problem of problems) {
    if (problem.status === "solved" || problem.prerequisites.length === 0) continue;
    const open = problem.prerequisites.flatMap((id) => {
      const prerequisite = byId.get(id);
      return prerequisite && prerequisite.status !== "solved" ? [prerequisite] : [];
    });
    if (open.length > 0) unmet.set(problem.id, open);
  }
  return unmet;
}

/**
 * The unmet prerequisites worth a remark in a path's own list: those that are
 * not earlier on the path. The order of the list already says "do these
 * first"; repeating it on every line would be noise.
 */
export function unmetOffPath(
  unmet: Map<string, ProblemSummary[]>,
  entries: PathEntry[][],
): Map<string, ProblemSummary[]> {
  const position = new Map(entries.flat().map((entry) => [entry.problem.id, entry.position]));
  const remaining = new Map<string, ProblemSummary[]>();
  for (const [id, prerequisites] of unmet) {
    const at = position.get(id);
    if (at === undefined) continue;
    const open = prerequisites.filter((prerequisite) => (position.get(prerequisite.id) ?? Infinity) > at);
    if (open.length > 0) remaining.set(id, open);
  }
  return remaining;
}

const ROMAN: [number, string][] = [
  [1000, "M"],
  [900, "CM"],
  [500, "D"],
  [400, "CD"],
  [100, "C"],
  [90, "XC"],
  [50, "L"],
  [40, "XL"],
  [10, "X"],
  [9, "IX"],
  [5, "V"],
  [4, "IV"],
  [1, "I"],
];

/** Roman numeral for a part number, the way a textbook numbers its parts. */
export function romanNumeral(value: number): string {
  let rest = Math.max(0, Math.floor(value));
  let out = "";
  for (const [amount, symbol] of ROMAN) {
    while (rest >= amount) {
      out += symbol;
      rest -= amount;
    }
  }
  return out;
}
