import { type ComponentType, type LazyExoticComponent, lazy } from "react";

/** What an interactive figure is given. It draws everything between the top of the figure and its caption. */
export interface InteractiveFigureProps {
  /** The caption as plain text: the accessible name of the drawing. */
  label: string;
}

/**
 * The interactive figures a statement can name with
 * `![caption](interactive:<name>)`, loaded on first use. A name is lowercase
 * words joined by hyphens. Each one is built in the app from shadcn primitives;
 * there are meant to be few. A test checks that every name used under
 * `problems/` is a key here.
 */
export const INTERACTIVE_FIGURES: Readonly<
  Record<string, LazyExoticComponent<ComponentType<InteractiveFigureProps>>>
> = {
  "quant-grid": lazy(() => import("./quant-grid")),
  "quant-grid-affine": lazy(() => import("./quant-grid-affine")),
  "rope-shift": lazy(() => import("./rope-shift")),
  "sampler-order": lazy(() => import("./sampler-order")),
};

/** The component for `name`, or null when the app has no such figure. */
export function interactiveFigure(name: string) {
  return Object.hasOwn(INTERACTIVE_FIGURES, name) ? (INTERACTIVE_FIGURES[name] ?? null) : null;
}

/** Every `interactive:<name>` a Markdown text names as an image, in order. */
export function interactiveNames(markdown: string): string[] {
  return [...markdown.matchAll(/!\[[^\]]*\]\(interactive:([^)\s]*)\)/g)].map((match) => match[1] ?? "");
}
