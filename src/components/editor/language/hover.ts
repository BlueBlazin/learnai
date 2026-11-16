import { LSPPlugin } from "@codemirror/lsp-client";
import type { Extension } from "@codemirror/state";
import { type EditorView, hoverTooltip, type Tooltip } from "@codemirror/view";
import type { EditorDocs } from "@/lib/api/types";
import { markupText } from "@/lib/editor/completion";
import { hasDocContent, hoverContent } from "@/lib/editor/doc-card";
import { isUnknownSignature, parseHover } from "@/lib/editor/doc-format";
import { type DocsLookup, mayBeLibraryName, needsDocsFallback } from "@/lib/editor/docs-lookup";
import { dottedNameAt } from "@/lib/editor/expression";
import { docCard, fillDocCard } from "./doc-view";

/** The same pause the app's tooltips wait before opening. */
const HOVER_DELAY_MS = 350;
/** How long a hover waits for the docs helper before showing what the server gave. */
const FALLBACK_WAIT_MS = 250;

interface ServerHover {
  contents: unknown;
  range?: { start: { line: number; character: number }; end: { line: number; character: number } };
}

function hoverSource(docs: DocsLookup) {
  return async (view: EditorView, pos: number): Promise<Tooltip | null> => {
    const plugin = LSPPlugin.get(view);
    if (!plugin) return null;
    const line = view.state.doc.lineAt(pos);
    const expression = dottedNameAt(line.text, pos - line.from);
    if (expression === null) return null;

    plugin.client.sync();
    let result: ServerHover | null = null;
    try {
      result = await plugin.client.request<unknown, ServerHover | null>("textDocument/hover", {
        textDocument: { uri: plugin.uri },
        position: plugin.toPosition(pos),
      });
    } catch {
      // No answer from the server; the docs helper may still know a dotted name.
    }
    const text = result ? markupText(result.contents) : null;
    const hover = text ? parseHover(text) : null;

    // The server has no docstring for names that only get one at import time: ask the library.
    let docsAnswer: EditorDocs | null = null;
    let pending: Promise<EditorDocs | null> | null = null;
    const source = view.state.doc.toString();
    if ((!hover || needsDocsFallback(hover.doc)) && mayBeLibraryName(source, expression)) {
      const lookup = docs.lookup(source, expression);
      const nothingYet = !hover || isUnknownSignature(hover.signature);
      const early = await Promise.race([
        lookup.then((found) => ({ found })),
        // With nothing else to show, the answer is worth waiting for.
        ...(nothingYet
          ? []
          : [new Promise<null>((resolve) => setTimeout(() => resolve(null), FALLBACK_WAIT_MS))]),
      ]);
      if (early) docsAnswer = early.found;
      else pending = lookup;
    }

    const content = hoverContent(hover, docsAnswer);
    if (!hasDocContent(content)) return null;

    let from = pos;
    let to = pos;
    try {
      if (result?.range) {
        from = plugin.fromPosition(result.range.start);
        to = plugin.fromPosition(result.range.end);
      } else {
        const word = view.state.wordAt(pos);
        if (word) ({ from, to } = word);
      }
    } catch {
      // The document changed under the request; the card closes on the next edit anyway.
    }
    return {
      pos: from,
      end: to,
      above: true,
      create(target) {
        const dom = docCard(target, content, "cm-doc-card-hover");
        let live = true;
        void pending?.then((found) => {
          if (!live || !found) return;
          fillDocCard(dom, target, hoverContent(hover, found));
          target.requestMeasure();
        });
        return {
          dom,
          destroy() {
            live = false;
          },
        };
      },
    };
  };
}

/**
 * Hover documentation: the signature and the docstring of the name under the
 * pointer, from the language server, completed by the docs helper when the
 * server has no docstring. Own source on the client's request API: the
 * built-in one renders Markdown as HTML and cannot be extended.
 */
export function hoverDocs(docs: DocsLookup): Extension {
  return hoverTooltip(hoverSource(docs), {
    hoverTime: HOVER_DELAY_MS,
    hideOn: (transaction) => transaction.docChanged,
  });
}
