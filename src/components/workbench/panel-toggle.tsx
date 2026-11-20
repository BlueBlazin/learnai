import { PanelBottomClose, PanelBottomOpen, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { IconButton } from "@/components/app/icon-button";
import { SHORTCUTS } from "@/components/app/shortcut-hint";
import type { WorkspacePanel } from "@/lib/workspace-layout";

const PANEL = {
  statement: {
    name: "statement",
    shortcut: SHORTCUTS.toggleStatement,
    Hide: PanelLeftClose,
    Show: PanelLeftOpen,
  },
  results: {
    name: "results",
    shortcut: SHORTCUTS.toggleResults,
    Hide: PanelBottomClose,
    Show: PanelBottomOpen,
  },
} as const;

/**
 * Shows or hides one of the workspace's collapsible panels. The label and the
 * icon say what a press will do; `aria-expanded` says where the panel stands.
 * `panel` is also the id of the panel element it controls.
 */
export function PanelToggle({
  panel,
  collapsed,
  onToggle,
}: {
  panel: WorkspacePanel;
  collapsed: boolean;
  onToggle: () => void;
}) {
  const { name, shortcut, Hide, Show } = PANEL[panel];
  return (
    <IconButton
      label={`${collapsed ? "Show" : "Hide"} ${name}`}
      shortcut={shortcut}
      aria-expanded={!collapsed}
      aria-controls={panel}
      // A showing panel is the normal state, not a held-down button: no fill for it.
      className="aria-expanded:bg-transparent aria-expanded:hover:bg-muted dark:aria-expanded:hover:bg-muted/50"
      onClick={onToggle}
    >
      {collapsed ? <Show /> : <Hide />}
    </IconButton>
  );
}
