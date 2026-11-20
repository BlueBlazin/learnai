import { useCallback, useMemo, useRef, useState } from "react";
import { type PanelProps, type PanelSize, usePanelRef } from "react-resizable-panels";
import { isCollapsedSize } from "@/lib/workspace-layout";

export interface CollapsiblePanel {
  /** True while the panel is collapsed, however it got there: a toggle, a shortcut, or a drag. */
  collapsed: boolean;
  /** Shows the panel; does nothing when it is already showing. */
  expand: () => void;
  /** Hides the panel; does nothing when it is already hidden. */
  collapse: () => void;
  toggle: () => void;
  /** Spread onto the `ResizablePanel` this controls. */
  panelProps: Pick<
    PanelProps,
    "collapsible" | "collapsedSize" | "panelRef" | "elementRef" | "onResize" | "inert"
  >;
}

/**
 * Makes a `ResizablePanel` collapsible to nothing. The panel stays mounted, so
 * what it holds (scroll position, a run's results) is there when it comes back;
 * while collapsed it is `inert`, so nothing inside can take focus or be read.
 *
 * `collapsed` follows the panel's own size, so dragging the handle past the
 * panel's minimum and back is tracked like any other toggle.
 */
export function useCollapsiblePanel({
  initiallyCollapsed,
  restoreSize,
  onFocusLost,
}: {
  /** What the saved layout says, so the first frame already knows. */
  initiallyCollapsed: boolean;
  /**
   * The size (percent) the panel last showed at. The library forgets it after a
   * drag-collapse and across a restart, and would reopen the panel at its minimum.
   */
  restoreSize?: number;
  /** Called when collapsing hid the focused element, to put focus somewhere visible. */
  onFocusLost?: () => void;
}): CollapsiblePanel {
  const panelRef = usePanelRef();
  const elementRef = useRef<HTMLDivElement | null>(null);
  const [collapsed, setCollapsed] = useState(initiallyCollapsed);
  const latest = useRef({ restoreSize, onFocusLost });
  latest.current = { restoreSize, onFocusLost };

  const onResize = useCallback((size: PanelSize) => setCollapsed(isCollapsedSize(size.asPercentage)), []);

  const expand = useCallback(() => {
    const panel = panelRef.current;
    if (!panel?.isCollapsed()) return;
    panel.expand();
    const size = latest.current.restoreSize;
    if (size !== undefined && size > panel.getSize().asPercentage) panel.resize(`${size}%`);
  }, [panelRef]);

  const collapse = useCallback(() => {
    const panel = panelRef.current;
    if (!panel || panel.isCollapsed()) return;
    const heldFocus = elementRef.current?.contains(document.activeElement) ?? false;
    panel.collapse();
    if (heldFocus && panel.isCollapsed()) latest.current.onFocusLost?.();
  }, [panelRef]);

  const toggle = useCallback(() => {
    if (panelRef.current?.isCollapsed()) expand();
    else collapse();
  }, [panelRef, expand, collapse]);

  const panelProps = useMemo(
    () => ({ collapsible: true, collapsedSize: "0%", panelRef, elementRef, onResize, inert: collapsed }),
    [panelRef, onResize, collapsed],
  );

  return { collapsed, expand, collapse, toggle, panelProps };
}
