/**
 * The text cap of figures. A drawing scales with the pane and its lettering
 * with it; past this size the lettering stops growing while the shapes go on.
 * 14px is the interface's `text-base`.
 */
export const FIGURE_TEXT_CAP = 14;

/**
 * The largest font size, in the drawing's own units, that still renders at or
 * below the cap when a drawing `viewBoxWidth` units wide is shown
 * `renderedWidth` CSS pixels wide. A drawing without a viewBox is shown 1:1.
 * Null while the drawing has no width (it is hidden): there is nothing to cap.
 */
export function textCapUnits(
  renderedWidth: number,
  viewBoxWidth: number,
  cap: number = FIGURE_TEXT_CAP,
): number | null {
  if (!(renderedWidth > 0)) return null;
  const scale = viewBoxWidth > 0 ? renderedWidth / viewBoxWidth : 1;
  return cap / scale;
}

/** The custom property, set on the drawing's host, that carries `textCapUnits`. */
export const FIGURE_TEXT_MAX = "--figure-text-max";

/**
 * The CSS `font-size` of text drawn at `units`: its own size, or the cap when
 * that is smaller. Without a measured cap the text keeps its size.
 */
export function cappedFontSize(units: number): string {
  return `min(${units}px, var(${FIGURE_TEXT_MAX}, ${units}px))`;
}
