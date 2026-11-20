/**
 * The pure rules behind the workspace's collapsible panels: which panels can
 * collapse, what size counts as collapsed, which key toggles which panel, and
 * how a saved layout is reopened with a panel showing.
 */

/** The workspace panels that can be collapsed. The editor never is. */
export type WorkspacePanel = "statement" | "results";

/** A panel group's layout as react-resizable-panels saves it: percentages by panel id. */
export type PanelLayout = Record<string, number>;

/** Panels collapse to nothing; anything under this many percent is a collapsed panel. */
const COLLAPSED_BELOW = 0.5;

/** True when a panel of this size (a percentage of its group) is collapsed. */
export function isCollapsedSize(percentage: number | undefined): boolean {
  return percentage !== undefined && percentage < COLLAPSED_BELOW;
}

/**
 * The panel a key press toggles: Mod-B the statement, Mod-J the results.
 * `mod` is the platform's command key (Command on a Mac, Control elsewhere).
 */
export function panelForShortcut(event: {
  key: string;
  mod: boolean;
  shiftKey: boolean;
  altKey: boolean;
}): WorkspacePanel | null {
  if (!event.mod || event.shiftKey || event.altKey) return null;
  const key = event.key.toLowerCase();
  if (key === "b") return "statement";
  if (key === "j") return "results";
  return null;
}

/**
 * A saved layout with `panel` showing at `size` percent. A layout that already
 * shows the panel (or none at all) comes back unchanged; otherwise the other
 * panels give up the room in proportion to their sizes.
 */
export function withPanelExpanded<T extends PanelLayout | undefined>(
  layout: T,
  panel: string,
  size: number,
): T {
  if (!layout || !isCollapsedSize(layout[panel])) return layout;
  const others = Object.keys(layout).filter((id) => id !== panel);
  const total = others.reduce((sum, id) => sum + (layout[id] ?? 0), 0);
  if (total <= 0 || size <= 0 || size >= 100) return layout;
  const scale = (100 - size) / total;
  const next: PanelLayout = { [panel]: size };
  for (const id of others) next[id] = (layout[id] ?? 0) * scale;
  // Keep the saved key order: the library matches a layout to its panels by id.
  return Object.fromEntries(Object.keys(layout).map((id) => [id, next[id] ?? 0])) as T;
}
