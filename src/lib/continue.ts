import type { ProblemSummary, Submission, Track } from "@/lib/api";
import { type Curriculum, curriculumOrder, nextInArea, type Place, placeOf } from "@/lib/curriculum";
import { nextUnsolved } from "@/lib/filters";

export interface ContinueTarget {
  problem: ProblemSummary;
  /** Present when the suggestion is based on recent work rather than on the route or catalog order. */
  lastSubmission?: Submission;
  /** Where the problem sits in the curriculum, when it is in it. */
  place?: Place;
}

/**
 * What to open next. Prefers the most recently submitted problem that is still
 * unsolved (unfinished work comes first); otherwise it follows the first area:
 * the next unsolved core problem on its route, and once the core is finished
 * the next unsolved further problem in module order. When the area has nothing
 * left, the first unsolved problem in catalog order. Null when everything is
 * solved.
 */
export function pickContinue(
  problems: ProblemSummary[],
  tracks: Track[],
  submissions: Submission[],
  curriculum: Curriculum,
): ContinueTarget | null {
  const placed = (target: ContinueTarget): ContinueTarget => {
    const place = placeOf(curriculum, target.problem.id);
    return place ? { ...target, place } : target;
  };

  const byId = new Map(problems.map((problem) => [problem.id, problem]));
  for (const submission of submissions) {
    const problem = byId.get(submission.problemId);
    if (problem && problem.status !== "solved") return placed({ problem, lastSubmission: submission });
  }
  const area = curriculum.areas[0];
  const inArea = area ? nextInArea(area) : null;
  if (inArea) return placed({ problem: inArea.problem });
  const next = nextUnsolved(problems, tracks);
  return next ? placed({ problem: next }) : null;
}

/**
 * The problems before and after `id` in the curriculum's reading order: within
 * an area a module's core, then its further problems, then the next module.
 */
export function neighbors(
  curriculum: Curriculum,
  id: string,
): { previous: ProblemSummary | null; next: ProblemSummary | null } {
  const ordered = curriculumOrder(curriculum);
  const index = ordered.findIndex((problem) => problem.id === id);
  if (index === -1) return { previous: null, next: null };
  return { previous: ordered[index - 1] ?? null, next: ordered[index + 1] ?? null };
}
