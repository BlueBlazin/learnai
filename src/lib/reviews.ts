import type { ProblemSummary, Review, Track } from "@/lib/api";
import { pluralize } from "@/lib/format";

/** How many reviews a problem gets before it is mastered: one per interval. */
export const REVIEW_COUNT = 5;

const DAY_MS = 86_400_000;
const dayMonth = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" });
const dayMonthYear = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" });

/** "Review 2 of 5": the stage is 0-based, the count people read is not. */
export function reviewLabel(review: Pick<Review, "stage">): string {
  return `Review ${Math.min(review.stage + 1, REVIEW_COUNT)} of ${REVIEW_COUNT}`;
}

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** Whole local calendar days from the due date to today: 0 on the due day, negative before it. */
export function daysOverdue(dueAt: string, now: Date = new Date()): number {
  return Math.round((startOfDay(now) - startOfDay(new Date(dueAt))) / DAY_MS);
}

/** "due today", "1 day overdue", "3 days overdue": how late a due review is, by the calendar. */
export function overdueLabel(dueAt: string, now: Date = new Date()): string {
  const days = daysOverdue(dueAt, now);
  return days <= 0 ? "due today" : `${pluralize(days, "day")} overdue`;
}

/** A due date as a day: "today", "tomorrow", "Nov 18", or with the year when it is not this one. */
export function formatDueDay(dueAt: string, now: Date = new Date()): string {
  const due = new Date(dueAt);
  const days = -daysOverdue(dueAt, now);
  if (days <= 0) return "today";
  if (days === 1) return "tomorrow";
  return due.getFullYear() === now.getFullYear() ? dayMonth.format(due) : dayMonthYear.format(due);
}

/** The reviews that are due, soonest first (the order the core lists them in). */
export function dueReviews(reviews: Review[]): Review[] {
  return reviews.filter((review) => review.due);
}

/** The review to start next: the one that has been due longest. */
export function nextDueReview(reviews: Review[]): Review | null {
  return dueReviews(reviews)[0] ?? null;
}

/**
 * The quiet line for when nothing is due: how many reviews are scheduled and
 * when the next one comes. Null when there is nothing scheduled either.
 */
export function upcomingSummary(reviews: Review[], now: Date = new Date()): string | null {
  const upcoming = reviews.filter((review) => !review.due);
  const next = upcoming[0];
  if (!next) return null;
  const day = formatDueDay(next.dueAt, now);
  const when = day === "today" || day === "tomorrow" ? day : `on ${day}`;
  return upcoming.length === 1
    ? `1 review scheduled, due ${when}.`
    : `${pluralize(upcoming.length, "review")} scheduled; the next is due ${when}.`;
}

/** What an accepted review submit led to. */
export type ReviewOutcome =
  /** That was the last review: the problem is mastered. */
  | { kind: "mastered"; label: string }
  /** The schedule moved on. */
  | { kind: "advanced"; label: string; next: Review };

/**
 * Compares the schedule before an accepted review submit (`before`) with the
 * list read afterward. Null until the list reflects the submit, so a stale list
 * never claims an outcome.
 */
export function reviewOutcome(before: Review, after: Review[] | undefined): ReviewOutcome | null {
  if (after === undefined) return null;
  const label = reviewLabel(before);
  const next = after.find((review) => review.problemId === before.problemId);
  if (!next) return { kind: "mastered", label };
  if (next.stage === before.stage && next.lastReviewedAt === before.lastReviewedAt) return null;
  return { kind: "advanced", label, next };
}

/** The sentence for an outcome: "Review 2 of 5 done. The next review is due Nov 22." */
export function describeOutcome(outcome: ReviewOutcome, now: Date = new Date()): string {
  if (outcome.kind === "mastered") {
    return `${outcome.label} done. That was the last one: this problem is mastered.`;
  }
  const day = formatDueDay(outcome.next.dueAt, now);
  const when = day === "today" || day === "tomorrow" ? day : `on ${day}`;
  return `${outcome.label} done. The next review is due ${when}, in ${pluralize(outcome.next.intervalDays, "day")}.`;
}

/** A review with what a list needs to name it. */
export interface ReviewEntry {
  review: Review;
  title: string;
  /** The track's title, when the track is known. */
  track?: string;
}

/** Pairs reviews with their problems; a review whose problem is not in the catalog is left out. */
export function reviewEntries(reviews: Review[], problems: ProblemSummary[], tracks: Track[]): ReviewEntry[] {
  const byId = new Map(problems.map((problem) => [problem.id, problem]));
  const trackTitle = new Map(tracks.map((track) => [track.id, track.title]));
  return reviews.flatMap((review) => {
    const problem = byId.get(review.problemId);
    return problem ? [{ review, title: problem.title, track: trackTitle.get(problem.track) }] : [];
  });
}
