import { type ReactNode, useId, useLayoutEffect, useRef, useState } from "react";
import { Notice } from "@/components/app/notice";
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
  /** The drawing: `SvgFigureBody`, an image, or a `Notice` when it cannot be shown. */
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
 * An author's SVG, inlined so that `currentColor` and the figure palette
 * follow the theme. The source is sanitized first (shapes and text only; see
 * `svg-policy.ts`) and inserted as DOM nodes, never as markup.
 */
export function SvgFigureBody({ source, label }: { source: string; label: string }) {
  const host = useRef<HTMLDivElement>(null);
  // Ids inside the figure are prefixed so two figures never collide.
  const prefix = `fig${useId().replace(/[^A-Za-z0-9]/g, "")}-`;
  const [invalid, setInvalid] = useState(false);

  useLayoutEffect(() => {
    const svg = sanitizeSvg(source, prefix);
    host.current?.replaceChildren(...(svg ? [svg] : []));
    setInvalid(svg === null);
  }, [source, prefix]);

  return (
    <>
      <div ref={host} className="figure-svg" role="img" aria-label={label} hidden={invalid} />
      {invalid ? <Notice>This figure is not a valid SVG file.</Notice> : null}
    </>
  );
}
