import type { CSSProperties } from "react";
import type { Path, PathStage } from "@/lib/api";

/** The color a stage with none of its own falls back to. */
const FALLBACK = "var(--primary)";

/** Small marks (pips, swatches) must reach this contrast against white on the light theme. */
const MARK_CONTRAST = 3;

/** The stage's color as `#rrggbb`, or null when it has none. */
export function stageColor(stage: PathStage): string | null {
  return stage.color && /^#[0-9a-f]{6}$/i.test(stage.color) ? stage.color : null;
}

function parseHex(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function toHex(rgb: [number, number, number]): string {
  return `#${rgb.map((channel) => Math.round(channel).toString(16).padStart(2, "0")).join("")}`;
}

/** WCAG relative luminance of a `#rrggbb` color. */
export function relativeLuminance(hex: string): number {
  const [r, g, b] = parseHex(hex).map((channel) => {
    const value = channel / 255;
    return value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  }) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

/** WCAG contrast ratio between two `#rrggbb` colors, from 1 to 21. */
export function contrastRatio(a: string, b: string): number {
  const [light, dark] = [relativeLuminance(a), relativeLuminance(b)].sort((x, y) => y - x) as [
    number,
    number,
  ];
  return (light + 0.05) / (dark + 0.05);
}

/** `hex` with `share` (0 to 1) of black mixed in. */
function darken(hex: string, share: number): string {
  const [r, g, b] = parseHex(hex);
  const keep = 1 - share;
  return toHex([r * keep, g * keep, b * keep]);
}

/**
 * The color for small marks that must hold on light paper. On the dark theme it
 * is the color as is; on the light theme a color with under 3:1 contrast against
 * white is darkened, in 1 percent steps toward black, until it reaches 3:1.
 */
export function markColor(hex: string): string {
  if (contrastRatio(hex, "#ffffff") >= MARK_CONTRAST) return hex;
  for (let share = 0.01; share <= 1; share += 0.01) {
    const darkened = darken(hex, share);
    if (contrastRatio(darkened, "#ffffff") >= MARK_CONTRAST) return `light-dark(${darkened}, ${hex})`;
  }
  return `light-dark(#000000, ${hex})`;
}

/**
 * Custom properties for the element that wraps a screen showing the path's
 * stages, for stage N (1-based, the part number): `--stage-N` is the color at
 * full saturation, for bar fills on both themes, and `--stage-N-mark` is
 * `markColor`, for pips and swatches. A stage with no color gets the accent.
 */
export function stageColorVars(path: Path): CSSProperties {
  const vars: Record<string, string> = {};
  path.stages.forEach((stage, index) => {
    const hex = stageColor(stage);
    vars[`--stage-${index + 1}`] = hex ?? FALLBACK;
    vars[`--stage-${index + 1}-mark`] = hex ? markColor(hex) : FALLBACK;
  });
  return vars as CSSProperties;
}

/** The fill color of the stage at `index` (0-based), as a reference to the variable `stageColorVars` sets. */
export function stageFill(index: number): string {
  return `var(--stage-${index + 1})`;
}

/** The small-mark color of the stage at `index` (0-based). */
export function stageMark(index: number): string {
  return `var(--stage-${index + 1}-mark)`;
}
