import type { ProblemSummary } from "@/lib/api";

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
 * The unmet prerequisites worth a remark in a list that is itself a reading
 * order: those that do not come earlier in `order`. The order of the list
 * already says "do these first"; repeating it on every line would be noise.
 * A problem that is not in `order` gets no entry.
 */
export function unmetOutOfOrder(
  unmet: Map<string, ProblemSummary[]>,
  order: ProblemSummary[],
): Map<string, ProblemSummary[]> {
  const place = new Map<string, number>();
  for (const problem of order) {
    if (!place.has(problem.id)) place.set(problem.id, place.size);
  }
  const remaining = new Map<string, ProblemSummary[]>();
  for (const [id, prerequisites] of unmet) {
    const at = place.get(id);
    if (at === undefined) continue;
    const open = prerequisites.filter((prerequisite) => (place.get(prerequisite.id) ?? Infinity) > at);
    if (open.length > 0) remaining.set(id, open);
  }
  return remaining;
}
