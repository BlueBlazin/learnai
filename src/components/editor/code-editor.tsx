import { lazy, Suspense } from "react";
import { LoadingLines } from "@/components/app/states";
import type { CodeEditorProps } from "./code-editor-view";

const CodeEditorView = lazy(() => import("./code-editor-view").then((m) => ({ default: m.CodeEditorView })));

/**
 * The Python editor (CodeMirror 6). Controlled: `value` is the document and
 * `onChange` reports user edits. Four-space indentation, Tab indents, standard
 * history, search (Mod-F), and bracket handling. Run and Submit shortcuts are
 * handled by the workspace, not here. Given the `document` it holds, an
 * editable editor also offers suggestions, hover documentation, signature
 * help, and diagnostics from the app's language server.
 *
 * CodeMirror loads on first use, so screens without an editor never pay for it.
 */
export function CodeEditor(props: CodeEditorProps) {
  return (
    <Suspense fallback={<LoadingLines lines={6} className="p-4" />}>
      <CodeEditorView {...props} />
    </Suspense>
  );
}
