import { useEffect, useRef } from "react";
import { isModKey } from "@/lib/platform";
import { panelForShortcut } from "@/lib/workspace-layout";

/**
 * Mod-B shows or hides the statement; Mod-J the results. Captured at the
 * window, like Run and Submit, so the shortcuts work from inside the editor.
 */
export function usePanelShortcuts(onToggleStatement: () => void, onToggleResults: () => void): void {
  const handlers = useRef({ onToggleStatement, onToggleResults });
  handlers.current = { onToggleStatement, onToggleResults };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const panel = panelForShortcut({
        key: event.key,
        mod: isModKey(event),
        shiftKey: event.shiftKey,
        altKey: event.altKey,
      });
      if (!panel || event.repeat) return;
      if (document.querySelector("[role=dialog], [role=alertdialog]")) return;
      event.preventDefault();
      event.stopPropagation();
      if (panel === "statement") handlers.current.onToggleStatement();
      else handlers.current.onToggleResults();
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () => window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, []);
}
