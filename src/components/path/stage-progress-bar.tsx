import { type PathEntry, stageCounts, stagesFinished } from "@/lib/paths";
import { stageFill } from "@/lib/stage-colors";

/**
 * Progress along a path as one bar of a segment per stage. A segment is as wide
 * as its stage has problems; its track is the stage color mixed into the paper
 * color, and the solved share is the color at full saturation, a whole problem
 * at a time. The colors come from `stageColorVars` on an ancestor.
 */
export function StageProgressBar({ title, entries }: { title: string; entries: PathEntry[][] }) {
  const counts = stageCounts(entries);
  const solved = counts.reduce((sum, stage) => sum + stage.solved, 0);
  const total = counts.reduce((sum, stage) => sum + stage.total, 0);
  return (
    <div
      role="progressbar"
      aria-label={`${title} progress`}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={solved}
      aria-valuetext={`${solved} of ${total} problems solved, ${stagesFinished(entries)} of ${entries.length} stages finished`}
      className="flex h-2.5 gap-0.75"
    >
      {counts.map((stage, index) => {
        const color = stageFill(index);
        return (
          <span
            // biome-ignore lint/suspicious/noArrayIndexKey: stages are positions along the path.
            key={index}
            className="h-full min-w-0 basis-0"
            style={{
              flexGrow: stage.total,
              backgroundColor: `color-mix(in oklab, ${color} 30%, var(--paper))`,
            }}
          >
            <span
              className="block h-full"
              style={{
                width: `${stage.total === 0 ? 0 : (stage.solved / stage.total) * 100}%`,
                backgroundColor: color,
              }}
            />
          </span>
        );
      })}
    </div>
  );
}
