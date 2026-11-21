import {
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
  useCallback,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { Notice } from "@/components/app/notice";
import { Button } from "@/components/ui/button";
import { findSteps, moveStep } from "./figure-steps";
import { cappedFontSize, FIGURE_TEXT_MAX, textCapUnits } from "./figure-text";
import { sanitizeSvg } from "./svg-sanitize";

/**
 * A numbered figure, set the way a textbook sets one: the drawing on the
 * measure and, beneath it, "Figure N." and the caption. Typography comes from
 * styles/prose.css (`.figure`), so it reads the same in a statement and alone.
 */
export function Figure({
  number,
  caption,
  children,
}: {
  /** Position among the figures of the document, from 1. */
  number: number;
  /** One sentence or two, set beneath the drawing after "Figure N." */
  caption?: ReactNode;
  /** The drawing: `SvgFigureBody`, `FigureDrawing`, an image, or a `Notice` when it cannot be shown. */
  children: ReactNode;
}) {
  return (
    <figure className="figure">
      {children}
      <figcaption>
        <span className="figure-label">Figure {number}.</span>
        {caption ? <> {caption}</> : null}
      </figcaption>
    </figure>
  );
}

/**
 * The text cap, in the one place every drawing goes through. The host of a
 * drawing carries `--figure-text-max`: the font size, in the drawing's units,
 * that renders at 14px at the current width. Text sized with `cappedFontSize`
 * stops growing there while the shapes keep scaling. The value follows the
 * pane through a ResizeObserver; the returned function measures again after
 * the drawing itself was replaced.
 */
function useFigureTextCap(host: RefObject<HTMLDivElement | null>): () => void {
  const measure = useCallback(() => {
    const node = host.current;
    if (!node) return;
    const svg = node.querySelector("svg");
    const max = svg
      ? textCapUnits(svg.getBoundingClientRect().width, svg.viewBox?.baseVal?.width ?? 0)
      : null;
    if (max === null) node.style.removeProperty(FIGURE_TEXT_MAX);
    else node.style.setProperty(FIGURE_TEXT_MAX, `${max.toFixed(3)}px`);
  }, [host]);

  useLayoutEffect(() => {
    const node = host.current;
    if (!node) return;
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    return () => observer.disconnect();
  }, [host, measure]);

  return measure;
}

/**
 * Puts the lettering of an inserted drawing under the cap: every `text` and
 * `tspan` gets its own size (as the author set it, on the element or on a
 * group above it) rewritten as `cappedFontSize`. Sizes are read before any is
 * written, so an inherited size is not read back capped.
 */
function capLettering(svg: SVGSVGElement): void {
  const nodes = [...svg.querySelectorAll<SVGElement>("text, tspan")];
  const sizes = nodes.map((node) => Number.parseFloat(getComputedStyle(node).fontSize));
  nodes.forEach((node, index) => {
    const size = sizes[index];
    if (size !== undefined && size > 0) node.style.fontSize = cappedFontSize(size);
  });
}

/**
 * The host of a drawing that the app draws itself (an interactive figure): one
 * `<svg>` with a viewBox as its child. It gets the same sizing and text cap as
 * an author's SVG; its text sets `font-size` with `cappedFontSize` or a class
 * that does (styles/prose.css).
 */
export function FigureDrawing({ label, children }: { label: string; children: ReactNode }) {
  const host = useRef<HTMLDivElement>(null);
  useFigureTextCap(host);
  return (
    <div ref={host} className="figure-svg" role="img" aria-label={label}>
      {children}
    </div>
  );
}

/**
 * An author's SVG, inlined so that `currentColor` and the figure palette
 * follow the theme. The source is sanitized first (shapes and text only; see
 * `svg-policy.ts`) and inserted as DOM nodes, never as markup.
 *
 * A file with step groups (`figure-steps.ts`) is shown one frame at a time,
 * with Previous and Next, a count, and the note of the step beneath. The left
 * and right arrow keys step while the figure or one of its buttons has focus.
 */
export function SvgFigureBody({ source, label }: { source: string; label: string }) {
  const host = useRef<HTMLDivElement>(null);
  const frames = useRef<SVGElement[]>([]);
  // Ids inside the figure are prefixed so two figures never collide.
  const prefix = `fig${useId().replace(/[^A-Za-z0-9]/g, "")}-`;
  const [invalid, setInvalid] = useState(false);
  const [notes, setNotes] = useState<readonly string[]>([]);
  const [step, setStep] = useState(0);
  const measure = useFigureTextCap(host);

  useLayoutEffect(() => {
    const svg = sanitizeSvg(source, prefix);
    host.current?.replaceChildren(...(svg ? [svg] : []));
    setInvalid(svg === null);
    const steps = svg ? findSteps(svg as unknown as { children: ArrayLike<SVGElement> }, prefix) : [];
    frames.current = steps.map((found) => found.element);
    setNotes((current) =>
      current.length === 0 && steps.length === 0 ? current : steps.map((found) => found.note),
    );
    setStep(0);
    if (svg) capLettering(svg);
    measure();
  }, [source, prefix, measure]);

  // `notes` changes whenever the frames do, so a new drawing gets its first frame.
  useLayoutEffect(() => {
    void notes;
    frames.current.forEach((frame, index) => {
      if (index === step) frame.removeAttribute("display");
      else frame.setAttribute("display", "none");
    });
  }, [step, notes]);

  const count = notes.length;
  const stepped = count > 0;
  const go = (delta: number) => setStep((current) => moveStep(current, delta, count));
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    go(event.key === "ArrowRight" ? 1 : -1);
  };

  return (
    // The same element with or without steps, so the inserted drawing is never remounted.
    <div
      className={stepped ? "figure-steps" : undefined}
      {...(stepped ? { role: "group", "aria-label": "Stepped figure", tabIndex: 0, onKeyDown } : undefined)}
    >
      <div ref={host} className="figure-svg" role="img" aria-label={label} hidden={invalid} />
      {invalid ? <Notice>This figure is not a valid SVG file.</Notice> : null}
      {stepped ? (
        <>
          <div className="figure-stepper">
            <Button
              variant="outline"
              size="sm"
              className="aria-disabled:pointer-events-none aria-disabled:opacity-50"
              disabled={step === 0}
              focusableWhenDisabled
              onClick={() => go(-1)}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="aria-disabled:pointer-events-none aria-disabled:opacity-50"
              disabled={step === count - 1}
              focusableWhenDisabled
              onClick={() => go(1)}
            >
              Next
            </Button>
            <span className="figure-step-count">
              Step {step + 1} of {count}
            </span>
          </div>
          <div className="figure-step-note" aria-live="polite">
            {notes[step]}
          </div>
        </>
      ) : null}
    </div>
  );
}
