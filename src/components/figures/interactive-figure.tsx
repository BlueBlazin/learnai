import { type ReactNode, Suspense } from "react";
import { Notice } from "@/components/app/notice";
import { LoadingLines } from "@/components/app/states";
import { Figure } from "@/components/prose/figure";
import { interactiveFigure } from "./registry";

/**
 * A figure the app draws itself, named in a statement as
 * `![caption](interactive:<name>)`. It is numbered and captioned like every
 * other figure; a name the app does not have shows a notice in its place.
 */
export function InteractiveFigure({
  name,
  number,
  alt,
  caption,
}: {
  name: string;
  number: number;
  /** The caption as plain text. */
  alt: string;
  caption: ReactNode;
}) {
  const Body = interactiveFigure(name);
  return (
    <Figure number={number} caption={caption}>
      {Body ? (
        <Suspense fallback={<LoadingLines lines={3} />}>
          <Body label={alt} />
        </Suspense>
      ) : (
        <Notice>
          There is no interactive figure named <code>{name}</code>.
        </Notice>
      )}
    </Figure>
  );
}
