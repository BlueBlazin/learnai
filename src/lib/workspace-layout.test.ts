import { describe, expect, it } from "vitest";
import { isCollapsedSize, panelForShortcut, withPanelExpanded } from "./workspace-layout";

describe("isCollapsedSize", () => {
  it("treats a zero-size panel as collapsed", () => {
    expect(isCollapsedSize(0)).toBe(true);
    expect(isCollapsedSize(0.001)).toBe(true);
  });

  it("treats a panel at or above its minimum as showing", () => {
    expect(isCollapsedSize(15)).toBe(false);
    expect(isCollapsedSize(46)).toBe(false);
  });

  it("does not call a panel collapsed before its size is known", () => {
    expect(isCollapsedSize(undefined)).toBe(false);
  });
});

describe("panelForShortcut", () => {
  const press = (
    key: string,
    modifiers: Partial<{ mod: boolean; shiftKey: boolean; altKey: boolean }> = {},
  ) => panelForShortcut({ key, mod: true, shiftKey: false, altKey: false, ...modifiers });

  it("maps Mod-B to the statement and Mod-J to the results", () => {
    expect(press("b")).toBe("statement");
    expect(press("j")).toBe("results");
  });

  it("ignores letter case, so Caps Lock does not break the shortcut", () => {
    expect(press("B")).toBe("statement");
    expect(press("J")).toBe("results");
  });

  it("needs the command key and no other modifier", () => {
    expect(press("b", { mod: false })).toBeNull();
    expect(press("b", { shiftKey: true })).toBeNull();
    expect(press("j", { altKey: true })).toBeNull();
  });

  it("leaves other keys alone", () => {
    expect(press("k")).toBeNull();
    expect(press("Enter")).toBeNull();
  });
});

describe("withPanelExpanded", () => {
  it("reopens a collapsed panel at the given size and shrinks the other to fit", () => {
    expect(withPanelExpanded({ statement: 0, work: 100 }, "statement", 46)).toEqual({
      statement: 46,
      work: 54,
    });
  });

  it("shares the room among several other panels in proportion", () => {
    expect(withPanelExpanded({ a: 0, b: 75, c: 25 }, "a", 20)).toEqual({ a: 20, b: 60, c: 20 });
  });

  it("keeps the saved order of panel ids", () => {
    expect(Object.keys(withPanelExpanded({ editor: 100, results: 0 }, "results", 40))).toEqual([
      "editor",
      "results",
    ]);
  });

  it("returns a layout that already shows the panel unchanged", () => {
    const layout = { statement: 40, work: 60 };
    expect(withPanelExpanded(layout, "statement", 46)).toBe(layout);
  });

  it("passes through a missing layout, an unknown panel, and a size that leaves no room", () => {
    expect(withPanelExpanded(undefined, "statement", 46)).toBeUndefined();
    const layout = { statement: 0, work: 100 };
    expect(withPanelExpanded(layout, "hints", 46)).toBe(layout);
    expect(withPanelExpanded(layout, "statement", 100)).toBe(layout);
  });
});
