import { useEffect, useSyncExternalStore } from "react";
import { backend } from "@/lib/api";
import { EditorSessionManager, type EditorSessionView } from "./session";

export type { EditorConnection, EditorSessionView } from "./session";
export type { EditorDocument } from "./uri";

/** The app's one editor session (the language server is a single process for the whole app). */
export const editorSession = new EditorSessionManager(backend);

// Leaving the page (a reload in development, the window closing) lets go of the server
// at once instead of after the linger. Best effort: core also stops it on exit and
// whenever another session starts.
if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => void editorSession.stop());
}

/** The session as the UI sees it; re-renders when it changes. */
export function useEditorSession(): EditorSessionView {
  return useSyncExternalStore(editorSession.subscribe, editorSession.getView);
}

/**
 * Keeps the session alive while the calling editor is mounted and `wanted`.
 * The first editor starts the server; it stops a little after the last one
 * leaves. Coming back to the window picks the server up again if another
 * window took it.
 */
export function useEditorSessionUser(wanted: boolean): void {
  useEffect(() => {
    if (!wanted) return;
    const release = editorSession.acquire();
    const resume = () => editorSession.resume();
    window.addEventListener("focus", resume);
    return () => {
      window.removeEventListener("focus", resume);
      release();
    };
  }, [wanted]);
}
