import { cn } from "cn";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Path } from "@/lib/api";
import { type PathEntry, romanNumeral, stageCounts } from "@/lib/paths";
import { sceneUrl } from "@/lib/scenes";
import { stageFill, stageMark } from "@/lib/stage-colors";

const COLUMNS = 5;

/** One square per problem in the stage: solved ones in the stage color, the rest quiet. */
function Pips({ index, total, solved }: { index: number; total: number; solved: number }) {
  return (
    <span aria-hidden className="flex gap-0.5">
      {Array.from({ length: total }, (_, pip) => (
        <span
          // biome-ignore lint/suspicious/noArrayIndexKey: the pips are positions, not items.
          key={pip}
          className={cn("size-1.5", pip >= solved && "bg-input")}
          style={pip < solved ? { backgroundColor: stageMark(index) } : undefined}
        />
      ))}
    </span>
  );
}

/**
 * The path, stage by stage, as plates in five columns: each stage's scene, always
 * in full color (a flat block of the stage color when it has none), with a pip
 * per problem and its name under it. The last stage takes the columns left over
 * in its row, so the grid closes with no hole. Progress is in the marks, not the
 * picture: a finished stage carries the solved mark and the stage under way has
 * its name in a heavier weight. What a stage covers is in its tooltip. The
 * colors come from `stageColorVars` on an ancestor.
 */
export function StagePlates({
  path,
  entries,
  onOpen,
}: {
  path: Path;
  entries: PathEntry[][];
  /** Opens the Path screen at the stage with this part number (1-based). */
  onOpen: (number: number) => void;
}) {
  const counts = stageCounts(entries);
  const current = counts.findIndex((stage) => stage.solved < stage.total);
  const count = path.stages.length;
  // The last plate fills out its row: one column when the stages divide evenly.
  const lastSpan = COLUMNS - ((count - 1) % COLUMNS);
  return (
    <ol className="grid grid-cols-5 gap-x-4 gap-y-8">
      {path.stages.map((stage, index) => {
        const { solved, total } = counts[index] ?? { solved: 0, total: stage.problems.length };
        const numeral = romanNumeral(index + 1);
        const last = index === count - 1;
        const wide = last && lastSpan > 1;
        const scene = sceneUrl(stage.scene);
        const summary = [stage.place, stage.description].filter(Boolean).join(". ");
        return (
          <li
            key={stage.title}
            aria-current={index === current ? "step" : undefined}
            className="flex min-w-0 flex-col items-start"
            style={last ? { gridColumn: `span ${lastSpan} / span ${lastSpan}` } : undefined}
          >
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    variant="ghost"
                    aria-label={`Part ${numeral}, ${stage.title}: ${solved} of ${total} solved`}
                    className="h-30 w-full cursor-pointer overflow-hidden rounded-sm border-0 p-0"
                    style={scene ? undefined : { backgroundColor: stageFill(index) }}
                    onClick={() => onOpen(index + 1)}
                  />
                }
              >
                {scene ? (
                  <img
                    src={scene}
                    alt=""
                    draggable={false}
                    // A wide last plate is a strip: keep the top of its picture in the crop.
                    className={cn(
                      "block size-full object-cover",
                      wide ? "object-[center_22%]" : "object-center",
                    )}
                  />
                ) : null}
              </TooltipTrigger>
              {summary ? <TooltipContent side="top">{summary}</TooltipContent> : null}
            </Tooltip>
            <span className="mt-3 flex h-1.5 items-center">
              <Pips index={index} total={total} solved={solved} />
            </span>
            <span className="tabular mt-2.5 flex h-4 items-center gap-1.5 text-xs text-muted-foreground">
              {numeral}
              {total > 0 && solved === total ? <StatusMark status="solved" /> : null}
            </span>
            <span className={cn("text-sm text-balance", index === current && "font-semibold")}>
              {stage.title}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
