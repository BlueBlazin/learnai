import { lazy, Suspense } from "react";
import { LoadingLines } from "@/components/app/states";
import type { ProseProps } from "./prose-content";

const ProseContent = lazy(() => import("./prose-content").then((m) => ({ default: m.ProseContent })));

/**
 * Renders authored Markdown (statements, hints, editorials) as typeset prose:
 * GFM tables, KaTeX math, highlighted listings, and note/tip/warning callouts.
 * All typography comes from styles/prose.css.
 *
 * The Markdown, math, and highlighting libraries load on first use, so screens
 * without prose never pay for them.
 */
export function Prose(props: ProseProps) {
  return (
    <Suspense fallback={<LoadingLines lines={6} className="max-w-measure" />}>
      <ProseContent {...props} />
    </Suspense>
  );
}
