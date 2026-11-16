import type { Framework } from "@/lib/api/types";

/** The file an editor holds: a problem's solution in one framework, or its review attempt. */
export interface EditorDocument {
  problemId: string;
  framework: Framework;
  /** The review attempt (`<problem>/review/<framework>.py`), not the saved solution. */
  review?: boolean;
}

/**
 * The URI the language server knows a document by (docs/ARCHITECTURE.md section
 * 4.4). Built from the `rootUri` of the `ready` state, never from a file path:
 * the server reports under the symlink-resolved root.
 */
export function documentUri(rootUri: string, document: EditorDocument): string {
  const segments = [
    document.problemId,
    ...(document.review ? ["review"] : []),
    `${document.framework}.py`,
  ].map(encodeURIComponent);
  return `${rootUri.replace(/\/+$/, "")}/${segments.join("/")}`;
}

/** A stable key for a document, for effects that must run when the document changes. */
export function documentKey(document: EditorDocument | null | undefined): string | null {
  if (!document) return null;
  return `${document.problemId}/${document.review ? "review/" : ""}${document.framework}`;
}
