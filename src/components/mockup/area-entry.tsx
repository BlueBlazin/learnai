import { AppLink } from "@/components/app/app-link";
import { pluralize } from "@/lib/format";
import { routes } from "@/lib/routes";
import { AreaBanner } from "./area-banner";
import type { Area } from "./model";
import { ModuleProgressBar } from "./module-progress-bar";

/**
 * An area on the list of areas: its banner, title, one sentence, its size in
 * plain counts, and the module bar. The whole block opens the area. The colors
 * come from `moduleColorVars` on an ancestor.
 */
export function AreaEntry({ area }: { area: Area }) {
  return (
    <article className="relative pb-12">
      <AreaBanner modules={area.modules} className="mb-6" />
      <h2 className="selectable max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
        <AppLink
          to={routes.devAreasMockup("area")}
          tone="quiet"
          stretched
          // The button's pressed nudge is a transform, which would shrink the stretched hit area to the title mid-click.
        >
          {area.title}
        </AppLink>
      </h2>
      <p className="mt-3 max-w-measure font-serif text-md text-pretty text-muted-foreground">
        {area.description}
      </p>
      <p className="tabular mt-3 text-sm text-muted-foreground">
        {pluralize(area.modules.length, "module")}, {pluralize(area.problemCount, "problem")},{" "}
        {area.coreCount} core
      </p>
      <div className="mt-5">
        <ModuleProgressBar title={area.title} modules={area.modules} />
      </div>
    </article>
  );
}
