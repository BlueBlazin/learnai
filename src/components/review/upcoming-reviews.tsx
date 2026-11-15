import { cn } from "cn";
import type { Review } from "@/lib/api";
import { upcomingSummary } from "@/lib/reviews";

/**
 * The quiet line for when no review is due: how many are scheduled and when the
 * next one comes. Renders nothing when nothing is scheduled.
 */
export function UpcomingReviews({
  reviews,
  now,
  className,
}: {
  reviews: Review[];
  now?: Date;
  className?: string;
}) {
  const summary = upcomingSummary(reviews, now);
  if (!summary) return null;
  return <p className={cn("selectable text-sm text-muted-foreground", className)}>{summary}</p>;
}
