import type { EditorStatus } from "@/lib/api/types";
import type { EditorSessionView } from "./session";

/** What Settings says about suggestions, from the stored status and the live session. */
export type SuggestionsState =
  | { kind: "off" }
  | { kind: "notInstalled" }
  | { kind: "installed" }
  | { kind: "starting" }
  | { kind: "running" }
  | { kind: "restarting" }
  | { kind: "failed"; message: string };

/**
 * Reduces the two sources to one line of status. The live session wins while
 * it has something to say; otherwise the stored status does (which knows
 * whether the server is installed without starting it).
 */
export function suggestionsState(status: EditorStatus, session: EditorSessionView): SuggestionsState {
  if (!status.enabled) return { kind: "off" };
  if (!status.installed) return { kind: "notInstalled" };
  switch (session.status) {
    case "starting":
      return { kind: "starting" };
    case "ready":
      return { kind: "running" };
    case "restarting":
      return { kind: "restarting" };
    case "failed":
      return { kind: "failed", message: session.message };
    default:
      return { kind: "installed" };
  }
}

/** The version to show: the running server's, or the installed one. */
export function suggestionsVersion(status: EditorStatus, session: EditorSessionView): string | null {
  const running = session.status === "ready" ? session.connection.serverVersion : null;
  return running ?? status.serverVersion;
}
