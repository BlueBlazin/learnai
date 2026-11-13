import { ProblemTable } from "@/components/problem/problem-table";
import type { ProblemSummary, Track } from "@/lib/api";

/**
 * One track in the library: a heading that stays pinned while its problems
 * scroll under it, a one-line description, and the problem table.
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
      <header className="sticky top-0 z-10 flex items-baseline gap-4 border-b border-rule bg-paper pt-4 pb-2">
        <h2 id={`track-${track.id}`} className="shrink-0 font-serif text-lg font-medium">
          {track.title}
        </h2>
        <p className="min-w-0 flex-1 truncate text-xs text-muted-foreground">{track.description}</p>
        <p className="tabular shrink-0 text-xs text-muted-foreground">
          {filtered
            ? `${problems.length} of ${track.problemCount} shown`
            : `${track.solvedCount} of ${track.problemCount} solved`}
        </p>
      </header>
      <ProblemTable problems={problems} label={`${track.title} problems`} />
    </section>
  );
}
