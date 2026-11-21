/**
 * Stepped figures. One SVG file can hold frames: groups directly under the
 * root with ids `step-1`, `step-2`, and so on, each with a `<desc>` child that
 * is the note of that step. Whatever is outside the step groups shows in every
 * frame. This module finds the frames; `SvgFigureBody` shows one at a time.
 */

/** The part of an element this module reads, so the parsing runs without a DOM. */
export interface StepNode {
  localName: string;
  id: string;
  textContent: string | null;
  children: ArrayLike<StepNode>;
}

export interface FigureStep<E extends StepNode> {
  /** The group that holds this frame. */
  element: E;
  /** The text of the group's `<desc>`, on one line; empty when there is none. */
  note: string;
}

/** The number in `step-<n>`, or null. `prefix` is what the sanitizer put before every id. */
export function stepNumber(id: string, prefix = ""): number | null {
  if (!id.startsWith(prefix)) return null;
  const match = /^step-([1-9]\d*)$/.exec(id.slice(prefix.length));
  return match ? Number(match[1]) : null;
}

/**
 * The frames of a figure in order: `step-1`, `step-2`, ... up to the first
 * number that is missing. Empty when the figure has no `step-1`, which makes
 * it an ordinary figure. Only groups directly under the root count.
 */
export function findSteps<E extends StepNode>(
  root: { children: ArrayLike<E> },
  prefix = "",
): FigureStep<E>[] {
  const byNumber = new Map<number, E>();
  for (const child of Array.from(root.children)) {
    if (child.localName !== "g") continue;
    const number = stepNumber(child.id, prefix);
    if (number !== null && !byNumber.has(number)) byNumber.set(number, child);
  }
  const steps: FigureStep<E>[] = [];
  for (let number = 1; byNumber.has(number); number++) {
    const element = byNumber.get(number) as E;
    const desc = Array.from(element.children).find((child) => child.localName === "desc");
    steps.push({ element, note: (desc?.textContent ?? "").replace(/\s+/g, " ").trim() });
  }
  return steps;
}

/** The step that `delta` moves to from `current`, kept inside `0 .. count - 1`. */
export function moveStep(current: number, delta: number, count: number): number {
  return Math.max(0, Math.min(count - 1, current + delta));
}
