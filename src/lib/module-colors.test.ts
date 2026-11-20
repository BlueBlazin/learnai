import { describe, expect, it } from "vitest";
import {
  contrastRatio,
  markColor,
  moduleColor,
  moduleColorVars,
  moduleFill,
  moduleMark,
  swatchColor,
} from "./module-colors";

describe("contrastRatio", () => {
  it("matches the WCAG reference pairs", () => {
    expect(contrastRatio("#000000", "#ffffff")).toBeCloseTo(21, 5);
    expect(contrastRatio("#ffffff", "#ffffff")).toBeCloseTo(1, 5);
    expect(contrastRatio("#777777", "#ffffff")).toBeCloseTo(4.48, 2);
    expect(contrastRatio("#ffffff", "#777777")).toBeCloseTo(4.48, 2);
  });
});

describe("markColor", () => {
  it("returns a color that already clears 3:1 unchanged", () => {
    expect(markColor("#2a71cc")).toBe("#2a71cc");
  });

  it("darkens a pale yellow to at least 3:1 on the light theme and keeps it as is on the dark", () => {
    const hex = "#f8d034";
    const result = markColor(hex);
    const match = /^light-dark\((#[0-9a-f]{6}), (#[0-9a-f]{6})\)$/.exec(result);
    expect(match).not.toBeNull();
    expect(match?.[2]).toBe(hex);
    expect(contrastRatio(match?.[1] ?? "", "#ffffff")).toBeGreaterThanOrEqual(3);
    // Only as dark as it needs to be: one step lighter would not clear.
    expect(contrastRatio(hex, "#ffffff")).toBeLessThan(3);
  });
});

describe("moduleColor", () => {
  it("returns the hex, or null when the track has none or it is malformed", () => {
    expect(moduleColor({ color: "#ed9b11" })).toBe("#ed9b11");
    expect(moduleColor({})).toBeNull();
    expect(moduleColor({ color: "orange" })).toBeNull();
  });
});

describe("moduleColorVars", () => {
  const modules = [
    { number: 1, color: "#2a71cc" },
    { number: 2, color: null },
    { number: 3, color: "#f8d034" },
  ];

  it("sets a fill and a mark per module, by its number", () => {
    const vars = moduleColorVars(modules) as Record<string, string>;
    expect(vars["--module-1"]).toBe("#2a71cc");
    expect(vars["--module-1-mark"]).toBe("#2a71cc");
    expect(vars["--module-3"]).toBe("#f8d034");
    expect(vars["--module-3-mark"]).toMatch(/^light-dark\(/);
    expect(moduleFill(3)).toBe("var(--module-3)");
    expect(moduleMark(3)).toBe("var(--module-3-mark)");
  });

  it("falls back to the accent for a module with no color", () => {
    const vars = moduleColorVars(modules) as Record<string, string>;
    expect(vars["--module-2"]).toBe("var(--primary)");
    expect(vars["--module-2-mark"]).toBe("var(--primary)");
    expect(swatchColor(null)).toBe("var(--primary)");
    expect(swatchColor("#f8d034")).toBe(markColor("#f8d034"));
  });
});
