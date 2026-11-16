import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import { editorSession } from "@/lib/editor";
import { backend } from "./index";
import { keys } from "./queries";

export interface EditorInstallState {
  status: "idle" | "running" | "succeeded" | "failed";
  log: string[];
  message: string | null;
}

const idle: EditorInstallState = { status: "idle", log: [], message: null };

/**
 * Installs the language server behind suggestions and collects its streamed
 * log, the way `useEnvInstall` does for a framework. A successful install
 * starts a session for the editors that are waiting.
 */
export function useEditorInstall() {
  const client = useQueryClient();
  const [state, setState] = useState<EditorInstallState>(idle);

  const install = useCallback(async () => {
    setState({ status: "running", log: [], message: null });
    try {
      await backend.editorInstall((event) => {
        if (event.type === "log") {
          setState((s) => ({ ...s, log: [...s.log, event.text] }));
        } else {
          setState((s) => ({ ...s, status: event.ok ? "succeeded" : "failed", message: event.message }));
          void client.invalidateQueries({ queryKey: keys.editorStatus });
          void client.invalidateQueries({ queryKey: keys.env });
          if (event.ok) editorSession.restart();
        }
      });
    } catch (error) {
      setState((s) => ({
        ...s,
        status: "failed",
        message: error instanceof Error ? error.message : String(error),
      }));
    }
  }, [client]);

  return { state, install };
}
