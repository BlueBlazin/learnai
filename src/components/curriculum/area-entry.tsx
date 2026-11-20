import { AppLink } from "@/components/app/app-link";
import type { CurriculumArea } from "@/lib/curriculum";
import { pluralize } from "@/lib/format";
import { moduleColorVars } from "@/lib/module-colors";
import { routes } from "@/lib/routes";
import { AreaBanner } from "./area-banner";
import { ModuleProgressBar } from "./module-progress-bar";

/** An area's size in plain counts: "15 modules, 213 problems, 84 core". */
export function areaCounts(area: CurriculumArea): string {
  const size = `${pluralize(area.modules.length, "module")}, ${pluralize(area.problemCount, "problem")}`;
  return area.coreCount > 0 ? `${size}, ${area.coreCount} core` : size;
}

/**
 * An area on the list of areas: its banner, title, one sentence, its size in
 * plain counts, and the module bar. The whole block opens the area. It sets
 * its own module colors, so areas can be listed one after another.
 */
export function AreaEntry({ area }: { area: CurriculumArea }) {
  return (
    <article className="relative pb-12" style={moduleColorVars(area.modules)}>
      <AreaBanner modules={area.modules} className="mb-6" />
      <h2 className="selectable max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
        <AppLink to={routes.area(area.id)} tone="quiet" stretched>
          {area.title}
        </AppLink>
      </h2>
      {area.description ? (
        <p className="mt-3 max-w-measure font-serif text-md text-pretty text-muted-foreground">
          {area.description}
        </p>
      ) : null}
      <p className="tabular mt-3 text-sm text-muted-foreground">{areaCounts(area)}</p>
      <div className="mt-5">
        <ModuleProgressBar title={area.title} modules={area.modules} />
      </div>
    </article>
  );
}
