import { cn } from "cn";
import { useLayoutEffect, useRef, useState } from "react";
import { pluralize } from "@/lib/format";
import { type ActivityWeeks, activeDays, longestStreak } from "./mock-data";

const WEEKDAY_INITIALS = ["M", "T", "W", "T", "F", "S", "S"];

/** A cell is 14px with 4px between cells: one week takes 18px of width. */
const CELL = 14;
const GAP = 4;

/** A quiet neutral for a day with nothing, then four steps of the accent mixed into the paper color. */
const LEVEL_FILL = [
  "bg-accent",
  "bg-[color-mix(in_oklab,var(--primary)_28%,var(--paper))]",
  "bg-[color-mix(in_oklab,var(--primary)_52%,var(--paper))]",
  "bg-[color-mix(in_oklab,var(--primary)_76%,var(--paper))]",
  "bg-primary",
];

/**
 * Activity as a grid of small squares, a column per week and a row per
 * weekday, with one sentence under it that states the counts. It shows as many
 * of the latest weeks as fit the width it is given.
 */
export function ActivityCalendar({ weeks }: { weeks: ActivityWeeks }) {
  const grid = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState(22);
  useLayoutEffect(() => {
    const element = grid.current;
    if (!element) return;
    const measure = () => setFit(Math.max(1, Math.floor((element.clientWidth + GAP) / (CELL + GAP))));
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const shown = weeks.slice(-fit);
  const active = activeDays(shown);
  const longest = longestStreak(shown);
  const sentence = `${pluralize(active, "active day")} in the last ${shown.length} weeks. Longest streak: ${pluralize(longest, "day")}.`;
  return (
    <div className="flex flex-col gap-4">
      <div role="img" aria-label={sentence} className="flex gap-1.5">
        <div className="grid shrink-0 grid-rows-7 gap-1 text-2xs text-faint">
          {WEEKDAY_INITIALS.map((initial, index) => (
            <span
              // biome-ignore lint/suspicious/noArrayIndexKey: weekdays are positions; two share an initial.
              key={index}
              className="flex size-3.5 items-center justify-start leading-none"
            >
              {initial}
            </span>
          ))}
        </div>
        <div ref={grid} className="min-w-0 flex-1">
          <div className="grid w-fit grid-flow-col grid-rows-7 gap-1">
            {shown.flatMap((week, weekIndex) =>
              week.split("").map((day, dayIndex) => (
                <span
                  // biome-ignore lint/suspicious/noArrayIndexKey: the grid is fixed positions.
                  key={`${weekIndex}-${dayIndex}`}
                  className={cn("size-3.5", day === "-" ? "invisible" : LEVEL_FILL[Number(day)])}
                />
              )),
            )}
          </div>
        </div>
      </div>
      <p className="selectable text-sm text-pretty text-muted-foreground">{sentence}</p>
    </div>
  );
}
