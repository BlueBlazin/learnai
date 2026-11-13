import { cn } from "cn";
import { ProgressRule } from "@/components/problem/progress-rule";
import { Button } from "@/components/ui/button";
import type { Track } from "@/lib/api";

function TrackButton({
  title,
  solved,
  total,
  active,
  onSelect,
}: {
  title: string;
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
            "min-w-0 text-sm text-pretty whitespace-normal",
            active ? "font-semibold" : "font-medium",
          )}
        >
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

/** The library's track column: every track with its solved count, plus "All tracks". */
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
        title="All tracks"
        solved={solved}
        total={total}
        active={selected === null}
        onSelect={() => onSelect(null)}
      />
      {tracks.map((track) => (
        <TrackButton
          key={track.id}
          title={track.title}
          solved={track.solvedCount}
          total={track.problemCount}
          active={selected === track.id}
          onSelect={() => onSelect(track.id)}
        />
      ))}
    </nav>
  );
}
