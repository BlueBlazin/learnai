/** Which rows of a long, fixed-row-height list to render, and how much space the rest take. */
export interface WindowRange {
  /** Index of the first rendered row. */
  start: number;
  /** One past the index of the last rendered row. */
  end: number;
  /** Height in pixels standing in for the rows before `start`. */
  before: number;
  /** Height in pixels standing in for the rows from `end` on. */
  after: number;
}

/**
 * Computes the rows of a list that intersect a viewport, plus `overscan` rows
 * on each side so scrolling never shows a gap.
 *
 * `listTop` is the list's top edge relative to the viewport's top edge
 * (negative once the list has scrolled past it).
 */
export function windowRange({
  count,
  rowHeight,
  listTop,
  viewportHeight,
  overscan,
}: {
  count: number;
  rowHeight: number;
  listTop: number;
  viewportHeight: number;
  overscan: number;
}): WindowRange {
  if (count <= 0 || rowHeight <= 0) return { start: 0, end: 0, before: 0, after: 0 };
  const first = Math.floor(-listTop / rowHeight);
  const last = Math.ceil((viewportHeight - listTop) / rowHeight);
  const start = Math.min(count, Math.max(0, first - overscan));
  // Keep at least one row mounted so the row height can always be measured.
  const end = Math.max(Math.min(count, Math.max(0, last + overscan)), Math.min(count, start + 1));
  return { start, end, before: start * rowHeight, after: (count - end) * rowHeight };
}
