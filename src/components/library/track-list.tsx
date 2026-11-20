import { cn } from "cn";
import { ModuleSwatch } from "@/components/curriculum/module-swatch";
import { ProgressRule } from "@/components/problem/progress-rule";
import { Button } from "@/components/ui/button";
import type { Track } from "@/lib/api";
import { moduleColor } from "@/lib/module-colors";

function TrackButton({
  title,
  swatch,
  solved,
  total,
  active,
  onSelect,
}: {
  title: string;
  /** The track's module color as a swatch before its title: a hex, or null for the accent. */
  swatch?: string | null;
  solved: number;
  total: number;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <Button
      variant="ghost"
      aria-current={active ? "true" : undefined}
      className={cn(
        "h-auto w-full flex-col items-stretch gap-1.5 rounded-md px-2.5 py-2 text-left font-normal",
        active && "bg-accent",
      )}
      onClick={onSelect}
    >
      <span className="flex items-baseline justify-between gap-3">
        <span
          className={cn(
            "flex min-w-0 items-baseline gap-2 text-sm text-pretty whitespace-normal",
            active ? "font-semibold" : "font-medium",
          )}
        >
          {swatch !== undefined ? <ModuleSwatch color={swatch} className="relative -top-px" /> : null}
          {title}
        </span>
        <span className="tabular shrink-0 text-xs text-muted-foreground">
          {solved} / {total}
        </span>
      </span>
      <ProgressRule value={solved} total={total} label={`${title} progress`} showCount={false} />
    </Button>
  );
}

/** The library's track column: every track with its module swatch and solved count, plus "All tracks". */
export function TrackList({
  tracks,
  selected,
  onSelect,
}: {
  tracks: Track[];
  selected: string | null;
  onSelect: (trackId: string | null) => void;
}) {
  const solved = tracks.reduce((sum, track) => sum + track.solvedCount, 0);
  const total = tracks.reduce((sum, track) => sum + track.problemCount, 0);
  return (
    <nav aria-label="Tracks" className="flex flex-col gap-0.5">
      <TrackButton
        title="All modules"
        solved={solved}
        total={total}
        active={selected === null}
        onSelect={() => onSelect(null)}
      />
      {tracks.map((track) => (
        <TrackButton
          key={track.id}
          title={track.title}
          swatch={moduleColor(track)}
          solved={track.solvedCount}
          total={track.problemCount}
          active={selected === track.id}
          onSelect={() => onSelect(track.id)}
        />
      ))}
    </nav>
  );
}
