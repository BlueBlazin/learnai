import { cn } from "cn";
import { lazy, Suspense } from "react";
import type { CodeBlockProps } from "./code-block-content";

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
    <Suspense
      fallback={
        <figure className={cn("code-block rounded-md bg-code", props.className)}>
          <pre className="selectable overflow-x-auto px-4 py-3 font-mono text-code font-normal not-italic">
            <code className="block w-fit min-w-full">{props.code.replace(/\n$/, "")}</code>
          </pre>
        </figure>
      }
    >
      <CodeBlockContent {...props} />
    </Suspense>
  );
}
