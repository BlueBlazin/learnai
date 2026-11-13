import type { ProblemSummary, Submission, Track } from "@/lib/api";
import { nextUnsolved } from "@/lib/filters";

export interface ContinueTarget {
  problem: ProblemSummary;
  /** Present when the suggestion is based on recent work rather than catalog order. */
  lastSubmission?: Submission;
}

/**
 * What to open next. Prefers the most recently submitted problem that is still
 * unsolved; otherwise the first unsolved problem in catalog order; null when
 * everything is solved.
 */
export function pickContinue(
  problems: ProblemSummary[],
  tracks: Track[],
  submissions: Submission[],
): ContinueTarget | null {
  const byId = new Map(problems.map((problem) => [problem.id, problem]));
  for (const submission of submissions) {
    const problem = byId.get(submission.problemId);
    if (problem && problem.status !== "solved") return { problem, lastSubmission: submission };
  }
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
