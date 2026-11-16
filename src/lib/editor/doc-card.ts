import type { EditorDocs } from "@/lib/api/types";
import {
  type DocBlock,
  formatDoc,
  type HoverParts,
  isUnknownSignature,
  plainSignature,
  withoutLeadingSignature,
} from "./doc-format";
import { needsDocsFallback } from "./docs-lookup";

/** What a documentation card shows: hover, the completion side panel, and signature help share it. */
export interface DocCardContent {
  /** Code shown on top: a signature or a type. */
  signature?: string | null;
  /** The public dotted name, when it says more than the code does. */
  caption?: string | null;
  blocks?: DocBlock[];
  /** The documentation was cut short. */
  truncated?: boolean;
}

/** True when there is anything to show. */
export function hasDocContent(content: DocCardContent): boolean {
  return Boolean(content.signature || content.caption || content.blocks?.length);
}

/**
 * The hover card, from the language server's hover and the docs helper's
 * answer (either may be missing). The server's signature and documentation
 * win when it has them; the helper fills in what is missing, and adds the
 * public dotted name. A failed or empty lookup leaves what the server gave.
 */
export function hoverContent(hover: HoverParts | null, docs: EditorDocs | null): DocCardContent {
  const known = hover && !isUnknownSignature(hover.signature) ? hover.signature : null;
  const serverDoc = hover && !needsDocsFallback(hover.doc) ? hover.doc : null;
  return {
    signature: known ? plainSignature(known) : (docs?.signature ?? null),
    caption: docs?.name ?? null,
    blocks: serverDoc ? formatDoc(withoutLeadingSignature(serverDoc)) : docs?.doc ? formatDoc(docs.doc) : [],
    truncated: serverDoc ? false : (docs?.truncated ?? false),
  };
}

/**
 * The completion side panel for one item: its signature over its
 * documentation, the docs helper's when the server sent none.
 */
export function completionContent(
  detail: string | null,
  documentation: string | null,
  docs: EditorDocs | null,
): DocCardContent {
  const signature = detail ? plainSignature(detail) : null;
  if (!needsDocsFallback(documentation)) {
    return { signature, blocks: formatDoc(withoutLeadingSignature(documentation ?? "")) };
  }
  return {
    signature: signature ?? docs?.signature ?? null,
    caption: docs?.name ?? null,
    blocks: docs?.doc ? formatDoc(docs.doc) : [],
    truncated: docs?.truncated ?? false,
  };
}
