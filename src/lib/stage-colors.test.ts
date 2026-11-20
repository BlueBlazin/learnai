import { describe, expect, it } from "vitest";
import type { Path } from "@/lib/api";
import { contrastRatio, markColor, stageColor, stageColorVars } from "./stage-colors";

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

const stage = (color?: string) => ({
  title: "T",
  description: "",
  problems: [],
  ...(color ? { color } : {}),
});

describe("stageColor", () => {
  it("returns the hex, or null when the stage has none or it is malformed", () => {
    expect(stageColor(stage("#ed9b11"))).toBe("#ed9b11");
    expect(stageColor(stage())).toBeNull();
    expect(stageColor(stage("orange"))).toBeNull();
  });
});

describe("stageColorVars", () => {
  const path: Path = {
    id: "p",
    title: "P",
    description: "",
    stages: [stage("#2a71cc"), stage(), stage("#f8d034")],
    problemCount: 0,
    solvedCount: 0,
  };

  it("sets a fill and a mark per stage, numbered from 1", () => {
    const vars = stageColorVars(path) as Record<string, string>;
    expect(vars["--stage-1"]).toBe("#2a71cc");
    expect(vars["--stage-1-mark"]).toBe("#2a71cc");
    expect(vars["--stage-3"]).toBe("#f8d034");
    expect(vars["--stage-3-mark"]).toMatch(/^light-dark\(/);
  });

  it("falls back to the accent for a stage with no color", () => {
    const vars = stageColorVars(path) as Record<string, string>;
    expect(vars["--stage-2"]).toBe("var(--primary)");
    expect(vars["--stage-2-mark"]).toBe("var(--primary)");
  });
});
