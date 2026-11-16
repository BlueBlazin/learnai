import type { Extension } from "@codemirror/state";
import type { EditorConnection } from "@/lib/editor";
import { languageService } from "./client";
import { serverCompletion } from "./completion";
import { hoverDocs } from "./hover";
import { signatureHelp } from "./signature";

/**
 * Everything the language server adds to one editor for the document at
 * `uri`: completion with documentation, hover documentation, signature help,
 * and diagnostics. Opening the extension opens the document on the server
 * (`didOpen`); removing or replacing it closes it (`didClose`), so switching
 * problem or framework is a reconfiguration with a new URI.
 */
export function languageSupport(connection: EditorConnection, uri: string): Extension {
  const { client, docs } = languageService(connection);
  return [client.plugin(uri, "python"), serverCompletion(docs), hoverDocs(docs), signatureHelp()];
}
