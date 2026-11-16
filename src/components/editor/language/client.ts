import { LSPClient } from "@codemirror/lsp-client";
import { backend } from "@/lib/api";
import type { EditorConnection } from "@/lib/editor";
import { DocsLookup } from "@/lib/editor/docs-lookup";
import { serverDiagnostics } from "./diagnostics";

/** How long a request may take. The first completion in a large package indexes it. */
const REQUEST_TIMEOUT_MS = 8_000;

/** What the editors of one server process share: its LSP client and the docs they looked up. */
export interface LanguageService {
  readonly client: LSPClient;
  readonly docs: DocsLookup;
}

const services = new WeakMap<EditorConnection, LanguageService>();

/**
 * The LSP client for a connection, created (and `initialize` sent) the first
 * time an editor asks. Every connection is a fresh server process, so each
 * gets its own client; the previous one is dropped with its connection.
 */
export function languageService(connection: EditorConnection): LanguageService {
  let service = services.get(connection);
  if (!service) {
    const client = new LSPClient({
      rootUri: connection.rootUri,
      timeout: REQUEST_TIMEOUT_MS,
      extensions: [serverDiagnostics()],
      notificationHandlers: {
        // The server's own messages are for its log, not for a dialog over the code.
        "window/showMessage": () => true,
        "window/logMessage": () => true,
      },
    }).connect(connection.transport);
    // A server that dies before answering `initialize` leaves this rejected; the session reports it.
    client.initializing.catch(() => {});
    service = {
      client,
      docs: new DocsLookup((source, expr, member) => backend.editorDocs(source, expr, member)),
    };
    services.set(connection, service);
  }
  return service;
}
