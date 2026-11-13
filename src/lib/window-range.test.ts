import { describe, expect, it } from "vitest";
import { windowRange } from "./window-range";

const base = { count: 1000, rowHeight: 36, viewportHeight: 720, overscan: 5 };

describe("windowRange", () => {
  it("renders the first screen plus overscan at the top", () => {
    expect(windowRange({ ...base, listTop: 0 })).toEqual({
      start: 0,
      end: 25,
      before: 0,
      after: (1000 - 25) * 36,
    });
  });

  it("follows the scroll position and accounts for every row's height", () => {
    const range = windowRange({ ...base, listTop: -3600 });
    expect(range.start).toBe(95);
    expect(range.end).toBe(125);
    expect(range.before + (range.end - range.start) * 36 + range.after).toBe(1000 * 36);
  });

  it("clamps at the end of the list", () => {
    const range = windowRange({ ...base, listTop: -(1000 * 36 - 720) });
    expect(range.end).toBe(1000);
    expect(range.after).toBe(0);
    expect(range.start).toBe(975);
  });

  it("keeps one row mounted when the list is far off screen", () => {
    expect(windowRange({ ...base, listTop: 5000 })).toMatchObject({ start: 0, end: 1 });
    const below = windowRange({ ...base, listTop: -1_000_000 });
    expect(below).toMatchObject({ start: 1000, end: 1000, after: 0 });
  });

  it("handles a list shorter than the viewport and an empty list", () => {
    expect(windowRange({ ...base, count: 8, listTop: 40 })).toEqual({
      start: 0,
      end: 8,
      before: 0,
      after: 0,
    });
    expect(windowRange({ ...base, count: 0, listTop: 0 })).toEqual({ start: 0, end: 0, before: 0, after: 0 });
  });
});
