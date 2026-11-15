import { Notice } from "@/components/app/notice";
import { describeOutcome, type ReviewOutcome } from "@/lib/reviews";

/**
 * Under the verdict of an accepted review submit: when the next review is due,
 * or that the problem is mastered.
 */
export function ReviewOutcomeNote({ outcome, now }: { outcome: ReviewOutcome; now?: Date }) {
  return <Notice tone="pass">{describeOutcome(outcome, now)}</Notice>;
}
