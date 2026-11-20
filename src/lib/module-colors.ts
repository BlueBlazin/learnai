import type { CSSProperties } from "react";
import type { Track } from "@/lib/api";

/** The color a module with none of its own falls back to. */
const FALLBACK = "var(--primary)";

/** Small marks (pips, swatches) must reach this contrast against white on the light theme. */
const MARK_CONTRAST = 3;

/** The module's color as `#rrggbb`, or null when the track has none or it is malformed. */
export function moduleColor(track: Pick<Track, "color">): string | null {
  return track.color && /^#[0-9a-f]{6}$/i.test(track.color) ? track.color : null;
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

/** The small-mark color for a module's own color: `markColor`, or the accent when it has none. */
export function swatchColor(color: string | null): string {
  return color ? markColor(color) : FALLBACK;
}

/**
 * Custom properties for the element that wraps a screen showing an area's
 * modules, for module N (its 1-based number): `--module-N` is the color at
 * full saturation, for bar fills and flat plates on both themes, and
 * `--module-N-mark` is `markColor`, for pips. A module with no color gets the
 * accent.
 */
export function moduleColorVars(modules: { number: number; color: string | null }[]): CSSProperties {
  const vars: Record<string, string> = {};
  for (const module of modules) {
    vars[`--module-${module.number}`] = module.color ?? FALLBACK;
    vars[`--module-${module.number}-mark`] = swatchColor(module.color);
  }
  return vars as CSSProperties;
}

/** The fill color of module `number` (1-based), as a reference to the variable `moduleColorVars` sets. */
export function moduleFill(number: number): string {
  return `var(--module-${number})`;
}

/** The small-mark color of module `number` (1-based). */
export function moduleMark(number: number): string {
  return `var(--module-${number}-mark)`;
}
