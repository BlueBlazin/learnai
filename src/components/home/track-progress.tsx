import { AppLink } from "@/components/app/app-link";
import { RuledList } from "@/components/app/ruled-list";
import { ProgressRule } from "@/components/problem/progress-rule";
import type { Track } from "@/lib/api";
import { pluralize } from "@/lib/format";
import { routes } from "@/lib/routes";

/**
 * The tracks, each linking to itself in the library. `progress` shows solved
 * counts as rules; `contents` is the table of contents a new user sees: what
 * each track covers and how many problems it holds, with no row of zeros.
 */
export function TrackProgress({
  tracks,
  variant = "progress",
}: {
  tracks: Track[];
  variant?: "progress" | "contents";
}) {
  return (
    <RuledList framed={false}>
      {tracks.map((track) =>
        variant === "progress" ? (
          <li key={track.id} className="flex items-center gap-4 py-2">
            <span className="w-56 shrink-0 text-sm">
              <AppLink to={routes.library({ track: track.id })} tone="quiet">
                {track.title}
              </AppLink>
            </span>
            <ProgressRule
              value={track.solvedCount}
              total={track.problemCount}
              label={`${track.title} progress`}
              className="min-w-0 flex-1"
            />
          </li>
        ) : (
          <li key={track.id} className="flex flex-col gap-0.5 py-2.5 first:pt-0">
            <div className="flex items-baseline justify-between gap-4">
              <span className="font-serif text-md">
                <AppLink to={routes.library({ track: track.id })} tone="quiet">
                  {track.title}
                </AppLink>
              </span>
              <span className="tabular shrink-0 text-xs text-muted-foreground">
                {pluralize(track.problemCount, "problem")}
              </span>
            </div>
            {track.description ? (
              <p className="selectable text-xs text-pretty text-muted-foreground">{track.description}</p>
            ) : null}
          </li>
        ),
      )}
    </RuledList>
  );
}
