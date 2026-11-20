import { cn } from "cn";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { romanNumeral } from "@/lib/paths";
import { type MockStage, STAGES, stageColor, stageSolved } from "./mock-data";

/** One square per problem in the stage: solved ones in the stage color, the rest quiet. */
function Pips({ stage, solved }: { stage: MockStage; solved: number }) {
  return (
    <span aria-hidden className="flex gap-0.5">
      {stage.problems.map((id, index) => (
        <span
          key={id}
          className={cn("size-1.5", index >= solved && "bg-input")}
          style={index < solved ? { backgroundColor: stageColor(stage) } : undefined}
        />
      ))}
    </span>
  );
}

/**
 * The route, stage by stage, as plates in five columns: each stage's scene,
 * always in full color, with a pip per problem and its name under it. The last
 * stage, the summit, is three columns wide and closes the grid. Progress is in
 * the marks, not the picture: a finished stage carries the solved mark and the
 * stage under way has its name in a heavier weight. What a stage covers is in
 * its tooltip.
 */
export function RoutePlates({
  solved,
  onOpen,
}: {
  /** Problems solved along the path. */
  solved: number;
  onOpen: (stage: MockStage) => void;
}) {
  const current = STAGES.find((stage) => stageSolved(stage, solved) < stage.problems.length) ?? null;
  return (
    <ol className="grid grid-cols-5 gap-x-4 gap-y-8">
      {STAGES.map((stage) => {
        const count = stageSolved(stage, solved);
        const total = stage.problems.length;
        const numeral = romanNumeral(stage.number);
        const finished = count === total;
        const summit = stage.number === STAGES.length;
        return (
          <li
            key={stage.number}
            aria-current={stage === current ? "step" : undefined}
            className={cn("flex min-w-0 flex-col items-start", summit && "col-span-3")}
          >
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    variant="ghost"
                    aria-label={`Part ${numeral}, ${stage.title}: ${count} of ${total} solved`}
                    className="h-30 w-full cursor-pointer overflow-hidden rounded-sm border-0 p-0"
                    onClick={() => onOpen(stage)}
                  />
                }
              >
                <img
                  src={stage.scene}
                  alt=""
                  draggable={false}
                  // The summit plate is a wide strip: keep its sky, moon, and dome in the crop.
                  className={cn(
                    "block size-full object-cover",
                    summit ? "object-[center_22%]" : "object-center",
                  )}
                />
              </TooltipTrigger>
              <TooltipContent side="top">
                {stage.place}. {stage.description}
              </TooltipContent>
            </Tooltip>
            <span className="mt-3 flex h-1.5 items-center">
              <Pips stage={stage} solved={count} />
            </span>
            <span className="tabular mt-2.5 flex h-4 items-center gap-1.5 text-xs text-muted-foreground">
              {numeral}
              {finished ? <StatusMark status="solved" /> : null}
            </span>
            <span className={cn("text-sm text-balance", stage === current && "font-semibold")}>
              {stage.title}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
