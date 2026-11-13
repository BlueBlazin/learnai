import { lazy, Suspense } from "react";
import type { CodeBlockProps } from "./code-block-content";
import { CodeFrame } from "./code-frame";

const CodeBlockContent = lazy(() =>
  import("./code-block-content").then((m) => ({ default: m.CodeBlockContent })),
);

/**
 * A read-only code listing: statement examples, reference solutions, submission
 * snapshots. Highlighted with the same palette as the editor.
 *
 * The highlighter loads on first use; until then the code shows unhighlighted
 * in the same block, so nothing moves when the colors arrive.
 */
export function CodeBlock(props: CodeBlockProps) {
  return (
    <Suspense fallback={<CodeFrame className={props.className}>{props.code.replace(/\n$/, "")}</CodeFrame>}>
      <CodeBlockContent {...props} />
    </Suspense>
  );
}
