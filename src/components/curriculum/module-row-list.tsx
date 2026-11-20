import { AppLink } from "@/components/app/app-link";
import { RuledList } from "@/components/app/ruled-list";
import { type CurriculumModule, moduleCounts } from "@/lib/curriculum";
import { routes } from "@/lib/routes";
import { ModuleSwatch } from "./module-swatch";

/**
 * Modules as plain rows: the swatch, the title, what the module covers, and
 * its size. For the modules that belong to no area, under the area's plates.
 * The whole row opens the module.
 */
export function ModuleRowList({ modules, label }: { modules: CurriculumModule[]; label: string }) {
  return (
    <RuledList framed={false} aria-label={label}>
      {modules.map((module) => (
        <li
          key={module.track.id}
          className="relative flex items-baseline gap-8 px-2 py-3 transition-colors hover:bg-accent/60"
        >
          <h3 className="flex w-72 shrink-0 items-baseline gap-2.5 font-serif text-md">
            <ModuleSwatch color={module.color} className="self-center" />
            <AppLink
              to={routes.module(module.areaId, module.track.id)}
              tone="quiet"
              stretched
              className="min-w-0 shrink text-left"
            >
              {module.track.title}
            </AppLink>
          </h3>
          <p className="min-w-0 flex-1 text-sm text-pretty text-muted-foreground">
            {module.track.description}
          </p>
          <span className="tabular shrink-0 text-xs text-muted-foreground">{moduleCounts(module)}</span>
        </li>
      ))}
    </RuledList>
  );
}
