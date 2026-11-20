import { AppLink } from "@/components/app/app-link";
import { pluralize } from "@/lib/format";
import { routes } from "@/lib/routes";
import type { Area } from "./model";
import { ModuleProgressBar } from "./module-progress-bar";

/**
 * Where the learner stands in an area, as a sentence in the manner of
 * `PathSummary`: how many core problems are solved and which comes next, over
 * the module bar. Before anything is solved it gives the length and the
 * starting point instead of zeros.
 */
export function AreaSummary({ area }: { area: Area }) {
  const { next } = area;
  const link = next ? <AppLink to={routes.problem(next.problem.id)}>{next.problem.title}</AppLink> : null;
  return (
    <div className="flex flex-col gap-4 pb-10">
      <p className="selectable max-w-measure font-serif text-lg text-pretty">
        {area.coreCount === 0 ? (
          <>
            {pluralize(area.problemCount, "problem")} in {pluralize(area.modules.length, "module")}.
          </>
        ) : area.coreSolved === 0 ? (
          <>
            {pluralize(area.coreCount, "core problem")} in {pluralize(area.modules.length, "module")}.{" "}
            {link ? <span className="text-muted-foreground">It starts with {link}.</span> : null}
          </>
        ) : next ? (
          <>
            <span className="font-semibold tabular">{area.coreSolved}</span> of {area.coreCount} core problems
            solved.{" "}
            <span className="text-muted-foreground">
              Next is {link}, in{" "}
              <AppLink
                to={routes.devAreasMockup("module", next.module.track.id)}
                tone="quiet"
                className="text-muted-foreground"
              >
                {next.module.track.title}
              </AppLink>
              .
            </span>
          </>
        ) : (
          <>
            All {area.coreCount} core problems solved.{" "}
            <span className="text-muted-foreground">
              The core is finished; every module has further problems.
            </span>
          </>
        )}
      </p>
      <ModuleProgressBar title={area.title} modules={area.modules} />
    </div>
  );
}
