import { useEffect, useRef } from "react";
import { isModKey } from "@/lib/platform";

/**
 * Mod-Enter runs; Shift-Mod-Enter submits. Captured at the window so the
 * shortcuts work from the editor, the statement, or anywhere else in the workspace.
 */
export function useRunShortcuts(onRun: () => void, onSubmit: () => void): void {
  const handlers = useRef({ onRun, onSubmit });
  handlers.current = { onRun, onSubmit };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Enter" || !isModKey(event) || event.altKey) return;
      if (document.querySelector("[role=dialog], [role=alertdialog]")) return;
      event.preventDefault();
      event.stopPropagation();
      if (event.shiftKey) handlers.current.onSubmit();
      else handlers.current.onRun();
    };
    window.addEventListener("keydown", onKeyDown, { capture: true });
    return () => window.removeEventListener("keydown", onKeyDown, { capture: true });
  }, []);
}
