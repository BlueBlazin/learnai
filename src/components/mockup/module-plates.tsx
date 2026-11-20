import { cn } from "cn";
import { PixelIcon } from "@/components/app/pixel-icon";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { stageFill, stageMark } from "@/lib/stage-colors";
import { type Module, moduleCounts } from "./model";

/** One square per core problem: solved ones in the module color, the rest quiet. */
function Pips({ module }: { module: Module }) {
  return (
    <span aria-hidden className="flex gap-0.5">
      {module.core.map((problem, pip) => (
        <span
          key={problem.id}
          className={cn("size-1.5", pip >= module.coreSolved && "bg-input")}
          style={pip < module.coreSolved ? { backgroundColor: stageMark(module.index) } : undefined}
        />
      ))}
    </span>
  );
}

/**
 * An area's modules as plates in five columns, adapted from `StagePlates`: each
 * module's scene, whole and in full color (a flat block of the module color
 * when it has none), a pip per core problem, its number and name, and how many
 * core and further problems it holds. A module whose core is solved carries
 * the solved mark, and the star once every problem in it is solved; the module
 * holding the next core problem has its name in a heavier weight. What a
 * module covers is in its tooltip. The colors come from `moduleColorVars` on
 * an ancestor.
 */
export function ModulePlates({
  modules,
  currentId,
  onOpen,
}: {
  modules: Module[];
  /** The track id of the module that holds the next core problem. */
  currentId?: string | null;
  onOpen: (module: Module) => void;
}) {
  return (
    <ol className="grid grid-cols-5 gap-x-4 gap-y-8">
      {modules.map((module) => {
        const { track, core, further, coreSolved, furtherSolved } = module;
        const current = track.id === currentId;
        const coreDone = core.length > 0 && coreSolved === core.length;
        const total = core.length + further.length;
        const allDone = total > 0 && coreSolved + furtherSolved === total;
        return (
          <li
            key={track.id}
            aria-current={current ? "step" : undefined}
            className="flex min-w-0 flex-col items-start"
          >
            <Tooltip>
              <TooltipTrigger
                render={
                  <Button
                    variant="ghost"
                    aria-label={`Module ${module.numeral}, ${track.title}: ${moduleCounts(module)}`}
                    className="aspect-4/3 h-auto w-full cursor-pointer overflow-hidden rounded-sm border-0 p-0"
                    style={module.scene ? undefined : { backgroundColor: stageFill(module.index) }}
                    onClick={() => onOpen(module)}
                  />
                }
              >
                {module.scene ? (
                  <img
                    src={module.scene}
                    alt=""
                    draggable={false}
                    className="block size-full object-cover object-center"
                  />
                ) : null}
              </TooltipTrigger>
              {track.description ? <TooltipContent side="top">{track.description}</TooltipContent> : null}
            </Tooltip>
            <span className="mt-3 flex h-1.5 items-center">
              <Pips module={module} />
            </span>
            <span className="tabular mt-2.5 flex h-4 items-center gap-1.5 text-xs text-muted-foreground">
              {module.numeral}
              {coreDone ? <StatusMark status="solved" /> : null}
              {allDone ? <PixelIcon name="star" label="Every problem solved" /> : null}
            </span>
            <span className={cn("text-sm text-balance", current && "font-semibold")}>{track.title}</span>
            <span className="tabular mt-0.5 text-xs text-muted-foreground">{moduleCounts(module)}</span>
          </li>
        );
      })}
    </ol>
  );
}
