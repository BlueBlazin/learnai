import type { EditorDocs } from "@/lib/api/types";
import {
  type DocBlock,
  formatDoc,
  type HoverParts,
  isPlaceholderType,
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
 * True when a hover from the server leaves the documentation to the docs
 * helper: it has no docstring, or only the one of a placeholder type.
 */
export function hoverNeedsDocs(hover: HoverParts): boolean {
  return needsDocsFallback(hover.doc) || isPlaceholderType(hover.signature);
}

/** What the language server said about one completion item. */
export interface ServerItemDocs {
  /** The item's type or signature. */
  detail: string | null;
  documentation: string | null;
}

/**
 * A completion item's detail and documentation, without an answer that only
 * says the server does not know. For a name it cannot type (`torch.special.erf`,
 * bound at import time), the server sends the detail `Unknown` and, as the
 * documentation, the docstring of the `Unknown` type itself. Neither is about
 * the name, so both are dropped and the docs helper is asked instead.
 */
export function serverItemDocs(detail: string | null, documentation: string | null): ServerItemDocs {
  return isPlaceholderType(detail) ? { detail: null, documentation: null } : { detail, documentation };
}

/**
 * The hover card, from the language server's hover and the docs helper's
 * answer (either may be missing). The server's signature and documentation
 * win when it has them; the helper fills in what is missing, and adds the
 * public dotted name. A failed or empty lookup leaves what the server gave.
 */
export function hoverContent(hover: HoverParts | null, docs: EditorDocs | null): DocCardContent {
  const known = hover && !isUnknownSignature(hover.signature) ? hover.signature : null;
  const serverDoc = hover && hoverNeedsDocs(hover) ? null : (hover?.doc ?? null);
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
  serverDetail: string | null,
  serverDocumentation: string | null,
  docs: EditorDocs | null,
): DocCardContent {
  const { detail, documentation } = serverItemDocs(serverDetail, serverDocumentation);
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
