import { ProblemTable } from "@/components/problem/problem-table";
import type { ProblemSummary, Track } from "@/lib/api";

/**
 * One track in the library: a heading that stays pinned while its problems
 * scroll under it, the track's description in full, and the problem table.
 */
export function TrackSection({
  track,
  problems,
  filtered,
}: {
  track: Track;
  problems: ProblemSummary[];
  /** True when filters are hiding some of the track's problems. */
  filtered: boolean;
}) {
  return (
    <section aria-labelledby={`track-${track.id}`} className="pb-8">
      <header className="sticky top-0 z-10 flex items-baseline justify-between gap-4 border-b border-rule bg-paper pt-4 pb-2">
        <h2
          id={`track-${track.id}`}
          title={track.title}
          className="min-w-0 truncate font-serif text-lg font-medium"
        >
          {track.title}
        </h2>
        <p className="tabular shrink-0 text-xs text-muted-foreground">
          {filtered
            ? `${problems.length} of ${track.problemCount} shown`
            : `${track.solvedCount} of ${track.problemCount} solved`}
        </p>
      </header>
      {track.description ? (
        <p className="selectable max-w-measure pt-2.5 pb-1.5 text-sm text-pretty text-muted-foreground">
          {track.description}
        </p>
      ) : null}
      <ProblemTable problems={problems} label={`${track.title} problems`} />
    </section>
  );
}
