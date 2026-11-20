import { describe, expect, it } from "vitest";
import type { Submission } from "@/lib/api";
import {
  activeDayCount,
  activityByDay,
  calendarWeeks,
  firstActiveDay,
  longestStreak,
  solvedThisWeek,
  weekDays,
} from "./activity";

let nextId = 1;
function submit(
  problemId: string,
  at: string,
  over: Partial<Pick<Submission, "mode" | "verdict">> = {},
): Submission {
  return {
    id: nextId++,
    problemId,
    framework: "torch",
    mode: "submit",
    verdict: "accepted",
    passed: 1,
    total: 1,
    skipped: 0,
    durationMs: 1,
    target: "local",
    createdAt: new Date(at).toISOString(),
    review: false,
    ...over,
  };
}

// Local-time strings without a zone suffix, so the days are the local days on any machine.
const local = (iso: string) => new Date(iso);

describe("activityByDay", () => {
  it("counts accepted submits per local day and ignores everything else", () => {
    const byDay = activityByDay([
      submit("a", "2026-11-18T09:00:00"),
      submit("b", "2026-11-18T23:30:00"),
      submit("a", "2026-11-19T00:10:00"),
      submit("c", "2026-11-19T10:00:00", { verdict: "wrong_answer" }),
      submit("c", "2026-11-17T10:00:00", { mode: "run" }),
    ]);
    expect([...byDay.entries()]).toEqual([
      ["2026-11-18", 2],
      ["2026-11-19", 1],
    ]);
  });
});

describe("weekDays", () => {
  const byDay = new Map([
    ["2026-11-16", 1],
    ["2026-11-18", 2],
  ]);

  it("marks the active days of the current week and today's index", () => {
    const week = weekDays(byDay, local("2026-11-19T12:00:00"));
    expect(week.days).toEqual([true, false, true, false, false, false, false]);
    expect(week.today).toBe(3);
  });

  it("treats Monday as the first day and Sunday as the last", () => {
    expect(weekDays(byDay, local("2026-11-16T08:00:00")).today).toBe(0);
    const sunday = weekDays(byDay, local("2026-11-22T23:00:00"));
    expect(sunday.today).toBe(6);
    expect(sunday.days).toEqual([true, false, true, false, false, false, false]);
  });

  it("does not carry a week over: the Monday after a Sunday starts empty", () => {
    expect(weekDays(byDay, local("2026-11-23T08:00:00")).days).toEqual(Array(7).fill(false));
  });
});

describe("solvedThisWeek", () => {
  const now = local("2026-11-19T12:00:00");

  it("counts a problem once however often it was accepted this week", () => {
    expect(
      solvedThisWeek(
        [
          submit("a", "2026-11-16T09:00:00"),
          submit("a", "2026-11-18T09:00:00"),
          submit("b", "2026-11-19T09:00:00"),
        ],
        now,
      ),
    ).toBe(2);
  });

  it("ignores last week, runs, and failed attempts", () => {
    expect(
      solvedThisWeek(
        [
          submit("a", "2026-11-15T23:59:00"),
          submit("b", "2026-11-17T09:00:00", { mode: "run" }),
          submit("c", "2026-11-17T09:00:00", { verdict: "wrong_answer" }),
        ],
        now,
      ),
    ).toBe(0);
  });

  it("counts a week that spans a month boundary", () => {
    const boundary = local("2026-12-02T10:00:00"); // Wednesday; the week began on Nov 30
    expect(
      solvedThisWeek(
        [
          submit("a", "2026-11-30T08:00:00"),
          submit("b", "2026-12-01T08:00:00"),
          submit("c", "2026-11-29T08:00:00"),
        ],
        boundary,
      ),
    ).toBe(2);
  });

  it("includes Sunday of the current week when today is Sunday", () => {
    expect(solvedThisWeek([submit("a", "2026-11-22T08:00:00")], local("2026-11-22T20:00:00"))).toBe(1);
  });
});

describe("calendarWeeks", () => {
  it("ends with the current week, starts each week on Monday, and marks future days", () => {
    const byDay = new Map([["2026-11-19", 3]]);
    const weeks = calendarWeeks(byDay, 3, local("2026-11-19T12:00:00"));
    expect(weeks).toHaveLength(3);
    expect(weeks.every((week) => week.length === 7)).toBe(true);
    expect(weeks[0]?.[0]?.date).toBe("2026-11-02");
    expect(weeks[2]?.[0]?.date).toBe("2026-11-16");
    expect(weeks[2]?.[6]?.date).toBe("2026-11-22");
    expect(weeks[2]?.[3]).toEqual({ date: "2026-11-19", count: 3, future: false });
    expect(weeks[2]?.[4]).toEqual({ date: "2026-11-20", count: 0, future: true });
    expect(weeks[1]?.some((day) => day.future)).toBe(false);
  });

  it("runs across a month boundary", () => {
    const weeks = calendarWeeks(new Map(), 2, local("2026-12-02T09:00:00"));
    expect(weeks[0]?.map((day) => day.date)).toEqual([
      "2026-11-23",
      "2026-11-24",
      "2026-11-25",
      "2026-11-26",
      "2026-11-27",
      "2026-11-28",
      "2026-11-29",
    ]);
    expect(weeks[1]?.slice(0, 3).map((day) => day.date)).toEqual(["2026-11-30", "2026-12-01", "2026-12-02"]);
  });

  it("puts a Sunday and a Monday today in the right week", () => {
    const sunday = calendarWeeks(new Map(), 1, local("2026-11-22T09:00:00"));
    expect(sunday[0]?.[0]?.date).toBe("2026-11-16");
    expect(sunday[0]?.some((day) => day.future)).toBe(false);
    const monday = calendarWeeks(new Map(), 1, local("2026-11-23T09:00:00"));
    expect(monday[0]?.[0]?.date).toBe("2026-11-23");
    expect(monday[0]?.filter((day) => day.future)).toHaveLength(6);
  });
});

describe("streaks and counts", () => {
  const byDay = new Map([
    ["2026-10-30", 1],
    ["2026-10-31", 1],
    ["2026-11-01", 2],
    ["2026-11-03", 1],
    ["2026-11-04", 1],
  ]);

  it("finds the longest run of consecutive days across a month boundary", () => {
    expect(longestStreak(byDay)).toBe(3);
    expect(longestStreak(new Map())).toBe(0);
  });

  it("counts active days since a date, inclusive", () => {
    expect(activeDayCount(byDay, "2026-11-01")).toBe(3);
    expect(activeDayCount(byDay, "2026-10-01")).toBe(5);
  });

  it("finds the first active day", () => {
    expect(firstActiveDay(byDay)).toBe("2026-10-30");
    expect(firstActiveDay(new Map())).toBeNull();
  });
});
