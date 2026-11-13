import { cn } from "cn";
import type { SaveState } from "@/lib/api/use-solution-editor";

const TEXT: Record<SaveState, string> = {
  saved: "Saved",
  unsaved: "Unsaved changes",
  saving: "Saving",
  error: "Could not save",
};

/** Autosave status for the open solution file. Quiet unless saving failed. */
export function SaveIndicator({ state, className }: { state: SaveState; className?: string }) {
  return (
    <span
      role="status"
      className={cn("text-xs", state === "error" ? "font-medium text-fail" : "text-faint", className)}
    >
      {TEXT[state]}
    </span>
  );
}
