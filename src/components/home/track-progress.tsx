import { AppLink } from "@/components/app/app-link";
import { ProgressRule } from "@/components/problem/progress-rule";
import type { Track } from "@/lib/api";
import { routes } from "@/lib/routes";

/** Solved counts per track, each linking to that track in the library. */
export function TrackProgress({ tracks }: { tracks: Track[] }) {
  return (
    <ul className="flex flex-col">
      {tracks.map((track) => (
        <li key={track.id} className="flex items-center gap-4 border-b border-border/70 py-2 last:border-b-0">
          <AppLink
            to={routes.library({ track: track.id })}
            tone="quiet"
            className="w-56 shrink-0 truncate text-sm"
          >
            {track.title}
          </AppLink>
          <ProgressRule
            value={track.solvedCount}
            total={track.problemCount}
            label={`${track.title} progress`}
            className="min-w-0 flex-1"
          />
        </li>
      ))}
    </ul>
  );
}
