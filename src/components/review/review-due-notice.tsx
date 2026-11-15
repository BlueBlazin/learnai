import { Notice } from "@/components/app/notice";
import { Button } from "@/components/ui/button";
import type { Review } from "@/lib/api";
import { overdueLabel, reviewLabel } from "@/lib/reviews";

/**
 * Says, in the normal workspace of a solved problem, that its review is due
 * and offers to start it.
 */
export function ReviewDueNotice({
  review,
  disabled = false,
  onStart,
  now,
  layout = "strip",
}: {
  review: Review;
  disabled?: boolean;
  onStart: () => void;
  now?: Date;
  layout?: "strip" | "inline";
}) {
  return (
    <Notice
      tone="info"
      layout={layout}
      action={
        <Button variant="outline" size="xs" disabled={disabled} onClick={onStart}>
          Start review
        </Button>
      }
    >
      This problem is up for review ({reviewLabel(review).toLowerCase()}, {overdueLabel(review.dueAt, now)}).
    </Notice>
  );
}
