import {
  autocompletion,
  type Completion,
  type CompletionContext,
  type CompletionResult,
  type CompletionSource,
  snippet,
} from "@codemirror/autocomplete";
import { localCompletionSource } from "@codemirror/lang-python";
import { syntaxTree } from "@codemirror/language";
import { LSPPlugin } from "@codemirror/lsp-client";
import type { Extension } from "@codemirror/state";
import type { EditorView } from "@codemirror/view";
import { completionSortKey, completionValidity, kindWord, markupText } from "@/lib/editor/completion";
import { completionContent, type DocCardContent, hasDocContent, serverItemDocs } from "@/lib/editor/doc-card";
import { type DocsLookup, mayBeLibraryName, needsDocsFallback } from "@/lib/editor/docs-lookup";
import { receiverBefore } from "@/lib/editor/expression";
import { docCard } from "./doc-view";

/** The parts of an LSP completion item this source reads. */
interface ServerItem {
  label: string;
  kind?: number;
  detail?: string;
  documentation?: unknown;
  filterText?: string;
  insertText?: string;
  insertTextFormat?: number;
  textEdit?: { newText: string };
  textEditText?: string;
}

interface ServerList {
  isIncomplete?: boolean;
  items: ServerItem[];
  itemDefaults?: { insertTextFormat?: number };
}

const WORD = /[A-Za-z_][A-Za-z0-9_]*$/;
const SNIPPET = 2;

/** The documentation panel beside the list: the item's signature over its documentation. */
function infoCard(view: EditorView, content: DocCardContent): HTMLElement | null {
  return hasDocContent(content) ? docCard(view, content, "cm-doc-card-info") : null;
}

function toCompletion(
  item: ServerItem,
  list: ServerList,
  view: EditorView,
  docs: DocsLookup,
  receiver: string | null,
): Completion {
  const text = item.textEdit?.newText || item.textEditText || item.insertText || item.label;
  const format = item.insertTextFormat ?? list.itemDefaults?.insertTextFormat;
  const label = item.filterText || item.label;
  const completion: Completion = {
    label,
    displayLabel: item.label,
    detail: kindWord(item.kind),
    sortText: completionSortKey(item.label),
    info: () => {
      const { detail, documentation } = serverItemDocs(item.detail ?? null, markupText(item.documentation));
      const source = view.state.doc.toString();
      const name = receiver ? `${receiver}.${item.label}` : item.label;
      if (!needsDocsFallback(documentation) || receiver === null || !mayBeLibraryName(source, name)) {
        return infoCard(view, completionContent(detail, documentation, null));
      }
      // The server has the name but no docstring: ask the library itself, without holding up the list.
      return docs
        .lookup(source, receiver, item.label)
        .then((found) => infoCard(view, completionContent(detail, documentation, found)));
    },
  };
  if (format === SNIPPET) {
    const template = text.replace(/\\([$}\\])|\$(\d+)/g, (_all, escaped: string, field: string) =>
      escaped ? escaped : `\${${field}}`,
    );
    completion.apply = (target, option, from, to) => snippet(template)(target, option, from, to);
  } else if (text !== label) {
    completion.apply = text;
  }
  return completion;
}

/** True inside a string or a comment, where suggestions only get in the way of prose. */
function inProse(context: CompletionContext): boolean {
  const node = syntaxTree(context.state).resolveInner(context.pos, -1);
  return /String|Comment/.test(node.name);
}

/**
 * Completion from the language server: names in scope while typing a word,
 * members right after a dot, and anything on request (Ctrl-Space). Each item
 * says what kind of thing it is; resting on one shows its signature and
 * documentation, from the server or, when it has none, from the docs helper.
 *
 * Written on the client's request API rather than its built-in source, which
 * renders documentation as HTML and has no place for the fallback.
 */
function serverCompletionSource(docs: DocsLookup): CompletionSource {
  return async (context): Promise<CompletionResult | null> => {
    const view = context.view;
    const plugin = view && LSPPlugin.get(view);
    if (!view || !plugin) return localCompletionSource(context);
    const word = context.matchBefore(WORD);
    const from = word ? word.from : context.pos;
    const afterDot = context.state.sliceDoc(from - 1, from) === ".";
    if (!context.explicit && ((!word && !afterDot) || inProse(context))) return null;
    if (plugin.client.serverCapabilities && !plugin.client.serverCapabilities.completionProvider) {
      return localCompletionSource(context);
    }

    plugin.client.sync();
    const params = {
      textDocument: { uri: plugin.uri },
      position: plugin.toPosition(context.pos),
      context: afterDot && !word ? { triggerKind: 2, triggerCharacter: "." } : { triggerKind: 1 },
    };
    context.addEventListener("abort", () => plugin.client.cancelRequest(params));
    let result: ServerItem[] | ServerList | null;
    try {
      result = await plugin.client.request<typeof params, ServerItem[] | ServerList | null>(
        "textDocument/completion",
        params,
      );
    } catch {
      // Canceled, timed out, or the server went away: fall back to the names in this file.
      return context.aborted ? null : localCompletionSource(context);
    }
    if (!result) return null;
    const list: ServerList = Array.isArray(result) ? { items: result } : result;
    const line = context.state.doc.lineAt(from);
    const receiver = receiverBefore(line.text.slice(0, from - line.from));
    return {
      from,
      options: list.items.map((item) => toCompletion(item, list, view, docs, receiver)),
      validFor: completionValidity(
        context.state.sliceDoc(from, context.pos),
        list.isIncomplete === true,
        list.items.length,
      ),
    };
  };
}

/** The completion configuration while a language server is connected. */
export function serverCompletion(docs: DocsLookup): Extension {
  return autocompletion({ override: [serverCompletionSource(docs)] });
}
