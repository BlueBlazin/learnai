import type { Submission } from "@/lib/api";

/**
 * What the learner did, by local calendar day (the core's streak counts days the
 * same way). An active day has an accepted submit; runs and failed attempts do
 * not count.
 */

const DAY_MS = 86_400_000;

/** A local calendar day as `YYYY-MM-DD`. */
export function dayKey(date: Date): string {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

function parseDayKey(key: string): Date {
  const [year, month, day] = key.split("-").map(Number) as [number, number, number];
  return new Date(year, month - 1, day);
}

/** The local day `offset` days from `date`, at midnight. Safe across daylight saving changes. */
function addDays(date: Date, offset: number): Date {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate() + offset);
}

/** Midnight of the Monday that starts the week `date` is in. */
function mondayOf(date: Date): Date {
  return addDays(date, -((date.getDay() + 6) % 7));
}

/** Whole local days from `from` to `to` (both `YYYY-MM-DD`). */
export function daysBetween(from: string, to: string): number {
  return Math.round((parseDayKey(to).getTime() - parseDayKey(from).getTime()) / DAY_MS);
}

function isAcceptedSubmit(submission: Submission): boolean {
  return submission.mode === "submit" && submission.verdict === "accepted";
}

/** Accepted submits per local day. Days with none are absent. */
export function activityByDay(submissions: Submission[]): Map<string, number> {
  const byDay = new Map<string, number>();
  for (const submission of submissions) {
    if (!isAcceptedSubmit(submission)) continue;
    const key = dayKey(new Date(submission.createdAt));
    byDay.set(key, (byDay.get(key) ?? 0) + 1);
  }
  return byDay;
}

export interface CalendarDay {
  /** Local `YYYY-MM-DD`. */
  date: string;
  count: number;
  /** After today. */
  future: boolean;
}

/** The last `weeks` weeks ending with the current one, oldest first; a week is Monday to Sunday. */
export function calendarWeeks(byDay: Map<string, number>, weeks: number, now: Date): CalendarDay[][] {
  const today = dayKey(now);
  const first = addDays(mondayOf(now), -7 * (Math.max(1, weeks) - 1));
  return Array.from({ length: Math.max(1, weeks) }, (_, week) =>
    Array.from({ length: 7 }, (_, day) => {
      const date = dayKey(addDays(first, week * 7 + day));
      return { date, count: byDay.get(date) ?? 0, future: date > today };
    }),
  );
}

/** The current Monday-to-Sunday week: whether each day was active, and which index is today. */
export function weekDays(byDay: Map<string, number>, now: Date): { days: boolean[]; today: number } {
  const monday = mondayOf(now);
  return {
    days: Array.from({ length: 7 }, (_, day) => (byDay.get(dayKey(addDays(monday, day))) ?? 0) > 0),
    today: (now.getDay() + 6) % 7,
  };
}

/** How many different problems have an accepted submit this week. */
export function solvedThisWeek(submissions: Submission[], now: Date): number {
  const monday = dayKey(mondayOf(now));
  const sunday = dayKey(addDays(mondayOf(now), 6));
  const solved = new Set<string>();
  for (const submission of submissions) {
    if (!isAcceptedSubmit(submission)) continue;
    const key = dayKey(new Date(submission.createdAt));
    if (key >= monday && key <= sunday) solved.add(submission.problemId);
  }
  return solved.size;
}

/** The most days in a row with activity. */
export function longestStreak(byDay: Map<string, number>): number {
  const days = [...byDay.entries()]
    .filter(([, count]) => count > 0)
    .map(([key]) => key)
    .sort();
  let longest = 0;
  let run = 0;
  let previous: string | null = null;
  for (const day of days) {
    run = previous !== null && daysBetween(previous, day) === 1 ? run + 1 : 1;
    longest = Math.max(longest, run);
    previous = day;
  }
  return longest;
}

/** The number of active days on or after `sinceDate` (`YYYY-MM-DD`). */
export function activeDayCount(byDay: Map<string, number>, sinceDate: string): number {
  let count = 0;
  for (const [key, value] of byDay) if (value > 0 && key >= sinceDate) count += 1;
  return count;
}

/** The earliest active day, or null when there is none. */
export function firstActiveDay(byDay: Map<string, number>): string | null {
  let first: string | null = null;
  for (const [key, value] of byDay) if (value > 0 && (first === null || key < first)) first = key;
  return first;
}
