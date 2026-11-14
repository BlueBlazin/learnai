import type { Path, ProblemSummary, Submission, Track } from "@/lib/api";
import { nextUnsolved } from "@/lib/filters";
import { findOnPath, nextOnPath, pathEntries } from "@/lib/paths";

export interface ContinueTarget {
  problem: ProblemSummary;
  /** Present when the suggestion is based on recent work rather than on the path or catalog order. */
  lastSubmission?: Submission;
  /** Present when the problem is on the study path: which path, and where on it. */
  onPath?: { path: Path; position: number };
}

/**
 * What to open next. Prefers the most recently submitted problem that is still
 * unsolved (unfinished work comes first); otherwise the next unsolved problem
 * on the study path; otherwise, when there is no path or it is finished, a
 * problem already attempted, then the first unsolved problem in catalog order.
 * Null when everything is solved.
 *
 * `path` is the learner's study path (the first one the content root defines).
 */
export function pickContinue(
  problems: ProblemSummary[],
  tracks: Track[],
  submissions: Submission[],
  path?: Path | null,
): ContinueTarget | null {
  const entries = path ? pathEntries(path, problems) : [];
  const withPath = (target: ContinueTarget): ContinueTarget => {
    const entry = path ? findOnPath(entries, target.problem.id) : null;
    return path && entry ? { ...target, onPath: { path, position: entry.position } } : target;
  };

  const byId = new Map(problems.map((problem) => [problem.id, problem]));
  for (const submission of submissions) {
    const problem = byId.get(submission.problemId);
    if (problem && problem.status !== "solved") return withPath({ problem, lastSubmission: submission });
  }
  const onPath = nextOnPath(entries);
  if (onPath) return withPath({ problem: onPath.problem });
  const attempted = problems.find((problem) => problem.status === "attempted");
  if (attempted) return { problem: attempted };
  const next = nextUnsolved(problems, tracks);
  return next ? { problem: next } : null;
}

/** The problems before and after `id` in catalog order (track order, then problem order). */
export function neighbors(
  problems: ProblemSummary[],
  tracks: Track[],
  id: string,
): { previous: ProblemSummary | null; next: ProblemSummary | null } {
  const trackOrder = new Map(tracks.map((track) => [track.id, track.order]));
  const ordered = [...problems].sort(
    (a, b) => (trackOrder.get(a.track) ?? 0) - (trackOrder.get(b.track) ?? 0) || a.order - b.order,
  );
  const index = ordered.findIndex((problem) => problem.id === id);
  if (index === -1) return { previous: null, next: null };
  return { previous: ordered[index - 1] ?? null, next: ordered[index + 1] ?? null };
}
