import { describe, expect, it } from "vitest";
import type { ProblemSummary, Review, Track } from "@/lib/api";
import {
  daysOverdue,
  describeOutcome,
  dueReviews,
  formatDueDay,
  nextDueReview,
  overdueLabel,
  reviewEntries,
  reviewLabel,
  reviewOutcome,
  upcomingSummary,
} from "./reviews";

/** Local time, so calendar-day math does not depend on the machine's zone. */
const NOW = new Date(2026, 10, 15, 10, 0, 0);
const at = (days: number, hours = 0) => new Date(NOW.getTime() + days * 86_400_000 + hours * 3_600_000);

const review = (problemId: string, stage: number, due: Date, over: Partial<Review> = {}): Review => ({
  problemId,
  stage,
  intervalDays: [3, 7, 21, 60, 180][stage] ?? 180,
  dueAt: due.toISOString(),
  lastReviewedAt: null,
  due: due.getTime() <= NOW.getTime(),
  ...over,
});

describe("review labels", () => {
  it("counts reviews from one", () => {
    expect(reviewLabel({ stage: 0 })).toBe("Review 1 of 5");
    expect(reviewLabel({ stage: 4 })).toBe("Review 5 of 5");
    // A stage past the last interval never reads "6 of 5".
    expect(reviewLabel({ stage: 9 })).toBe("Review 5 of 5");
  });

  it("says how overdue a review is by the calendar, not by hours", () => {
    expect(overdueLabel(at(0, -1).toISOString(), NOW)).toBe("due today");
    // Nine hours ago was still today.
    expect(overdueLabel(at(0, -9).toISOString(), NOW)).toBe("due today");
    // Eleven hours ago was yesterday evening.
    expect(overdueLabel(at(0, -11).toISOString(), NOW)).toBe("1 day overdue");
    expect(overdueLabel(at(-3).toISOString(), NOW)).toBe("3 days overdue");
    expect(daysOverdue(at(2).toISOString(), NOW)).toBe(-2);
  });

  it("names a due day", () => {
    expect(formatDueDay(at(0, 5).toISOString(), NOW)).toBe("today");
    expect(formatDueDay(at(-4).toISOString(), NOW)).toBe("today");
    expect(formatDueDay(at(1).toISOString(), NOW)).toBe("tomorrow");
    expect(formatDueDay(at(3).toISOString(), NOW)).toBe("Nov 18");
    expect(formatDueDay(at(60).toISOString(), NOW)).toBe("Jan 14, 2027");
  });
});

describe("review lists", () => {
  const reviews = [
    review("old", 4, at(-3)),
    review("today", 1, at(0, -2)),
    review("soon", 0, at(2)),
    review("later", 2, at(19)),
  ];

  it("separates what is due and picks the longest overdue to start next", () => {
    expect(dueReviews(reviews).map((r) => r.problemId)).toEqual(["old", "today"]);
    expect(nextDueReview(reviews)?.problemId).toBe("old");
    expect(nextDueReview(reviews.slice(2))).toBeNull();
    expect(nextDueReview([])).toBeNull();
  });

  it("summarizes what is still to come", () => {
    expect(upcomingSummary(reviews, NOW)).toBe("2 reviews scheduled; the next is due on Nov 17.");
    expect(upcomingSummary([review("soon", 0, at(1))], NOW)).toBe("1 review scheduled, due tomorrow.");
    // Due reviews are not "upcoming", and nothing scheduled says nothing.
    expect(upcomingSummary(reviews.slice(0, 2), NOW)).toBeNull();
    expect(upcomingSummary([], NOW)).toBeNull();
  });

  it("pairs reviews with their problems and drops the ones the catalog does not know", () => {
    const problems = [
      { id: "old", title: "Stable Softmax", track: "foundations" },
      { id: "today", title: "Causal Mask", track: "gone" },
    ] as ProblemSummary[];
    const tracks = [{ id: "foundations", title: "Numerical Foundations" }] as Track[];
    const entries = reviewEntries(reviews, problems, tracks);
    expect(entries.map((entry) => [entry.review.problemId, entry.title, entry.track])).toEqual([
      ["old", "Stable Softmax", "Numerical Foundations"],
      ["today", "Causal Mask", undefined],
    ]);
  });
});

describe("the outcome of an accepted review submit", () => {
  const before = review("p", 1, at(-1), { lastReviewedAt: "2026-11-01T09:00:00.000Z" });

  it("waits for a list that reflects the submit", () => {
    expect(reviewOutcome(before, undefined)).toBeNull();
    // The list still holds the schedule as it was: stale, so no claim yet.
    expect(reviewOutcome(before, [before])).toBeNull();
  });

  it("reports the next review once the schedule has moved on", () => {
    const next = review("p", 2, at(21), { lastReviewedAt: NOW.toISOString() });
    const outcome = reviewOutcome(before, [review("other", 0, at(1)), next]);
    expect(outcome).toEqual({ kind: "advanced", label: "Review 2 of 5", next });
    expect(outcome && describeOutcome(outcome, NOW)).toBe(
      "Review 2 of 5 done. The next review is due on Dec 6, in 21 days.",
    );
  });

  it("reports mastery when the schedule is gone", () => {
    const last = review("p", 4, at(0, -1));
    const outcome = reviewOutcome(last, [review("other", 0, at(1))]);
    expect(outcome).toEqual({ kind: "mastered", label: "Review 5 of 5" });
    expect(outcome && describeOutcome(outcome, NOW)).toBe(
      "Review 5 of 5 done. That was the last one: this problem is mastered.",
    );
  });
});
