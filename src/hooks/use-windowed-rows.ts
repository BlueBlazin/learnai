import { type RefObject, useLayoutEffect, useState } from "react";
import { type WindowRange, windowRange } from "@/lib/window-range";

const SCROLL_VIEWPORT = "[data-slot=scroll-area-viewport]";

/**
 * Windows the rows of a long list that lives inside a `ScrollArea`: only the
 * rows near the visible part are rendered, and two spacers keep the scroll
 * height. Rows must share one height; it is measured from a rendered row
 * (marked `data-windowed-row`), starting from `estimatedRowHeight`.
 *
 * With `enabled` false every row is rendered and no listeners are attached.
 */
export function useWindowedRows(
  container: RefObject<HTMLElement | null>,
  count: number,
  {
    enabled = true,
    estimatedRowHeight,
    overscan = 12,
  }: { enabled?: boolean; estimatedRowHeight: number; overscan?: number },
): WindowRange {
  const everything: WindowRange = { start: 0, end: count, before: 0, after: 0 };
  const [range, setRange] = useState<WindowRange>(() =>
    enabled
      ? windowRange({ count, rowHeight: estimatedRowHeight, listTop: 0, viewportHeight: 0, overscan })
      : everything,
  );

  useLayoutEffect(() => {
    const element = container.current;
    if (!enabled || !element) return;
    const viewport = element.closest<HTMLElement>(SCROLL_VIEWPORT);

    // Runs straight from the scroll event: it only reads three rectangles, and the
    // state changes only when the range of rows does.
    const measure = () => {
      const row = element.querySelector<HTMLElement>("[data-windowed-row]");
      const rowHeight = row?.getBoundingClientRect().height || estimatedRowHeight;
      const bounds = viewport?.getBoundingClientRect();
      const next = windowRange({
        count,
        rowHeight,
        listTop: element.getBoundingClientRect().top - (bounds?.top ?? 0),
        viewportHeight: bounds?.height ?? window.innerHeight,
        overscan,
      });
      setRange((current) =>
        current.start === next.start &&
        current.end === next.end &&
        current.before === next.before &&
        current.after === next.after
          ? current
          : next,
      );
    };

    measure();
    const target: HTMLElement | Window = viewport ?? window;
    target.addEventListener("scroll", measure, { passive: true });
    window.addEventListener("resize", measure);
    // Content above the list (other tracks, filters) can grow or shrink.
    const observer = new ResizeObserver(measure);
    if (viewport?.firstElementChild) observer.observe(viewport.firstElementChild);
    return () => {
      target.removeEventListener("scroll", measure);
      window.removeEventListener("resize", measure);
      observer.disconnect();
    };
  }, [container, count, enabled, estimatedRowHeight, overscan]);

  if (!enabled) return everything;
  // The list can shrink (a filter) before the effect runs again.
  return range.end > count
    ? windowRange({ count, rowHeight: estimatedRowHeight, listTop: 0, viewportHeight: 0, overscan })
    : range;
}
