import { type RefObject, useLayoutEffect } from "react";

const SCROLL_VIEWPORT = "[data-slot=scroll-area-viewport]";

/** Scroll offsets by key, for as long as the window is open. */
const memory = new Map<string, number>();

/**
 * Remembers how far a `ScrollArea` was scrolled, so coming Back to a screen
 * returns to the same place instead of the top. `container` is the ScrollArea
 * root; `ready` is true once the content that gives it height has rendered.
 */
export function useScrollMemory(container: RefObject<HTMLElement | null>, key: string, ready: boolean): void {
  useLayoutEffect(() => {
    const viewport = container.current?.querySelector<HTMLElement>(SCROLL_VIEWPORT);
    if (!viewport || !ready) return;
    viewport.scrollTop = memory.get(key) ?? 0;
    const onScroll = () => memory.set(key, viewport.scrollTop);
    viewport.addEventListener("scroll", onScroll, { passive: true });
    return () => viewport.removeEventListener("scroll", onScroll);
  }, [container, key, ready]);
}
