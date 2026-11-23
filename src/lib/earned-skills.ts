import type { Skill } from "@/lib/api";
import type { RunState } from "@/lib/api/run-state";

/**
 * The stored submission of a run that can have earned a skill: an accepted
 * submit that has ended and is not a review attempt. Null for anything else,
 * including a run that is still going, a Run, and a submit that was not stored.
 */
export function earningSubmission(run: RunState): number | null {
  if (run.phase !== "done" || run.mode !== "submit" || run.review || run.verdict !== "accepted") {
    return null;
  }
  return run.submissionId;
}

export type EarnedAction =
  /** The skills a submit earned arrived. */
  | { type: "earned"; skills: Skill[] }
  /** The popup for the first skill was closed. */
  | { type: "next" }
  /** The rest are dropped. */
  | { type: "clear" };

/**
 * The skills waiting for their popup, the one showing first. Skills that arrive
 * while a popup is showing wait behind it; one already waiting is not added twice.
 */
export function earnedReducer(queue: Skill[], action: EarnedAction): Skill[] {
  switch (action.type) {
    case "earned": {
      const fresh = action.skills.filter((skill) => !queue.some((waiting) => waiting.id === skill.id));
      return fresh.length === 0 ? queue : [...queue, ...fresh];
    }
    case "next":
      return queue.slice(1);
    case "clear":
      return queue.length === 0 ? queue : [];
  }
}
