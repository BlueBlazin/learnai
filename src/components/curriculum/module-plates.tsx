import { cn } from "cn";
import { PixelIcon } from "@/components/app/pixel-icon";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { allSolved, type CurriculumModule, coreComplete, moduleCounts } from "@/lib/curriculum";
import { moduleFill, moduleMark } from "@/lib/module-colors";

/** One square per core problem: solved ones in the module color, the rest quiet. */
function Pips({ module }: { module: CurriculumModule }) {
  return (
    <span aria-hidden className="flex gap-0.5">
      {module.core.map(({ problem }, pip) => (
        <span
          key={problem.id}
          className={cn("size-1.5", pip >= module.coreSolved && "bg-input")}
          style={pip < module.coreSolved ? { backgroundColor: moduleMark(module.number) } : undefined}
        />
      ))}
    </span>
  );
}

/**
 * An area's modules as plates in five columns: each module's scene, whole and
 * in full color (a flat block of the module color when it has none), a pip per
 * core problem, its numeral and name, and how many core and further problems
 * it holds (left out when `compact`, on Home). Progress is in the marks, not
 * the picture: a module whose core is solved carries the solved mark, and the
 * star once every problem in it is solved. The module holding the next core
 * problem says "Up next" beside its numeral and has its name in a heavier
 * weight. What a module covers is in its tooltip. The colors come from
 * `moduleColorVars` on an ancestor.
 */
export function ModulePlates({
  modules,
  currentId,
  compact = false,
  onOpen,
}: {
  modules: CurriculumModule[];
  /** The track id of the module that holds the next core problem. */
  currentId?: string | null;
  /** Without the line of counts under each name. */
  compact?: boolean;
  onOpen: (module: CurriculumModule) => void;
}) {
  return (
    <ol className="grid grid-cols-5 gap-x-4 gap-y-8">
      {modules.map((module) => {
        const { track } = module;
        const current = track.id === currentId;
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
                    style={module.scene ? undefined : { backgroundColor: moduleFill(module.number) }}
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
            {/* As tall as the star, so a module that earns it keeps the same rows as its neighbors. */}
            <span className="tabular mt-1 flex h-6 items-center gap-1.5 text-xs text-muted-foreground">
              {module.numeral}
              {coreComplete(module) ? <StatusMark status="solved" /> : null}
              {allSolved(module) ? <PixelIcon name="star" label="Every problem solved" /> : null}
              {current ? <span className="ml-0.5 font-medium text-foreground">Up next</span> : null}
            </span>
            <span className={cn("text-sm text-balance", current && "font-semibold")}>{track.title}</span>
            {compact ? null : (
              <span className="tabular mt-0.5 text-xs text-muted-foreground">{moduleCounts(module)}</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
