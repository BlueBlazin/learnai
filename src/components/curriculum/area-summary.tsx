import { AppLink } from "@/components/app/app-link";
import type { CurriculumArea, RouteEntry } from "@/lib/curriculum";
import { pluralize } from "@/lib/format";
import { routes } from "@/lib/routes";
import { ModuleProgressBar } from "./module-progress-bar";

/**
 * Where the learner stands in an area, as a sentence over the module bar: how
 * many core problems are solved and, when `next` is given, which comes next
 * and in which module. Before anything is solved it gives the length and the
 * starting point instead of zeros. Home leaves `next` out: its headline
 * already names the problem. The bar's colors come from `moduleColorVars` on
 * an ancestor.
 */
export function AreaSummary({ area, next }: { area: CurriculumArea; next?: RouteEntry | null }) {
  const link = next ? <AppLink to={routes.problem(next.problem.id)}>{next.problem.title}</AppLink> : null;
  const modules = pluralize(area.modules.length, "module");
  return (
    <div className="flex flex-col gap-4">
      <p className="selectable max-w-measure font-serif text-lg text-pretty">
        {area.coreCount === 0 ? (
          <>
            {pluralize(area.problemCount, "problem")} in {modules}.
          </>
        ) : area.coreSolved === 0 ? (
          <>
            {pluralize(area.coreCount, "core problem")} in {modules}.{" "}
            {link ? <span className="text-muted-foreground">It starts with {link}.</span> : null}
          </>
        ) : area.coreSolved < area.coreCount ? (
          <>
            <span className="font-semibold tabular">{area.coreSolved}</span> of {area.coreCount} core problems
            solved.{" "}
            {next ? (
              <span className="text-muted-foreground">
                Next is {link}, in{" "}
                <AppLink
                  to={routes.module(area.id, next.module.track.id)}
                  tone="quiet"
                  className="text-muted-foreground"
                >
                  {next.module.track.title}
                </AppLink>
                .
              </span>
            ) : null}
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
