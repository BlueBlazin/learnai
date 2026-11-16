import { type Diagnostic, setDiagnostics } from "@codemirror/lint";
import { type LSPClientExtension, LSPPlugin } from "@codemirror/lsp-client";
import { ViewPlugin, type ViewUpdate } from "@codemirror/view";
import { type ServerDiagnostic, severityOf, visibleDiagnostics } from "@/lib/editor/diagnostics";
import { parseInline } from "@/lib/editor/doc-format";

const LABEL = { error: "Error", warning: "Warning", info: "Note" } as const;

/** The message as the tooltip shows it: what it is in a word, then the text with names in the code face. */
function renderMessage(severity: keyof typeof LABEL, message: string): () => Node {
  return () => {
    const fragment = document.createDocumentFragment();
    const label = document.createElement("span");
    label.className = "cm-diagnostic-label";
    label.textContent = LABEL[severity];
    const text = document.createElement("span");
    text.className = "cm-diagnostic-text";
    for (const part of parseInline(message)) {
      if (part.kind === "text") text.append(part.text);
      else {
        const code = document.createElement("code");
        code.className = "cm-doc-code-inline";
        code.textContent = part.text;
        text.append(code);
      }
    }
    fragment.append(label, text);
    return fragment;
  };
}

/** How long after the last edit the document is sent, so diagnostics follow typing without chasing it. */
const SYNC_DELAY_MS = 400;

const autoSync = ViewPlugin.fromClass(
  class {
    private pending: ReturnType<typeof setTimeout> | null = null;

    update(update: ViewUpdate) {
      if (!update.docChanged) return;
      if (this.pending) clearTimeout(this.pending);
      this.pending = setTimeout(() => {
        this.pending = null;
        LSPPlugin.get(update.view)?.client.sync();
      }, SYNC_DELAY_MS);
    }

    destroy() {
      if (this.pending) clearTimeout(this.pending);
    }
  },
);

/**
 * Diagnostics the server pushes, as quiet underlines: errors and warnings
 * only (hints are dropped), with the message on hover and nothing in the
 * gutter. Also keeps the server's copy of the document current while typing,
 * which is what makes it publish.
 */
export function serverDiagnostics(): LSPClientExtension {
  return {
    clientCapabilities: { textDocument: { publishDiagnostics: { versionSupport: true } } },
    notificationHandlers: {
      "textDocument/publishDiagnostics": (
        client,
        params: { uri: string; version?: number | null; diagnostics: ServerDiagnostic[] },
      ) => {
        const file = client.workspace.getFile(params.uri);
        // Diagnostics for an older version would be drawn over text that has moved on.
        if (!file || (params.version != null && params.version !== file.version)) return false;
        const view = file.getView();
        const plugin = view && LSPPlugin.get(view);
        if (!view || !plugin) return false;
        const length = view.state.doc.length;
        const diagnostics: Diagnostic[] = [];
        for (const item of visibleDiagnostics(params.diagnostics)) {
          try {
            const from = plugin.unsyncedChanges.mapPos(
              plugin.fromPosition(item.range.start, plugin.syncedDoc),
            );
            const to = plugin.unsyncedChanges.mapPos(plugin.fromPosition(item.range.end, plugin.syncedDoc));
            const severity = severityOf(item);
            diagnostics.push({
              from: Math.min(from, length),
              to: Math.min(Math.max(to, from), length),
              severity,
              message: item.message,
              renderMessage: renderMessage(severity, item.message),
            });
          } catch {
            // A range outside the document: the server was a step behind. The next push corrects it.
          }
        }
        view.dispatch(setDiagnostics(view.state, diagnostics));
        return true;
      },
    },
    editorExtension: autoSync,
  };
}
