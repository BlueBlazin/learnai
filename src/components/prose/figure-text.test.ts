import { describe, expect, it } from "vitest";
import { cappedFontSize, FIGURE_TEXT_CAP, textCapUnits } from "./figure-text";

/** The size in CSS pixels that text drawn at `units` renders at, as the browser resolves `cappedFontSize`. */
function rendered(units: number, renderedWidth: number, viewBoxWidth: number): number {
  const max = textCapUnits(renderedWidth, viewBoxWidth);
  const scale = renderedWidth / viewBoxWidth;
  return Math.min(units, max ?? units) * scale;
}

describe("figure text cap", () => {
  it("caps at 14 CSS pixels", () => {
    expect(FIGURE_TEXT_CAP).toBe(14);
  });

  it("gives the size in drawing units that renders at the cap", () => {
    // A 480-unit drawing on the 640px measure is at 4/3: 10.5 units are 14px.
    expect(textCapUnits(640, 480)).toBeCloseTo(10.5, 10);
    expect(textCapUnits(480, 480)).toBe(14);
    expect(textCapUnits(240, 480)).toBe(28);
    expect(textCapUnits(640, 480, 12)).toBeCloseTo(9, 10);
  });

  it("never lets text render above the cap, at any width", () => {
    for (let width = 200; width <= 640; width += 20) {
      for (const units of [9.5, 11, 13, 14, 16, 22]) {
        expect(rendered(units, width, 480)).toBeLessThanOrEqual(FIGURE_TEXT_CAP + 1e-9);
      }
    }
    expect(rendered(13, 640, 480)).toBeCloseTo(14, 10);
  });

  it("leaves smaller text alone, so it keeps scaling with the drawing", () => {
    expect(rendered(13, 460, 480)).toBeCloseTo((13 * 460) / 480, 10);
    expect(rendered(9.5, 640, 480)).toBeCloseTo((9.5 * 4) / 3, 10);
  });

  it("treats a drawing without a viewBox as shown 1:1, and skips a hidden one", () => {
    expect(textCapUnits(300, 0)).toBe(14);
    expect(textCapUnits(0, 480)).toBeNull();
    expect(textCapUnits(Number.NaN, 480)).toBeNull();
  });

  it("writes the size so that it falls back to the author's when no cap is measured", () => {
    expect(cappedFontSize(13)).toBe("min(13px, var(--figure-text-max, 13px))");
  });
});
