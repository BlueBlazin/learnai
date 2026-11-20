import { AppLink } from "@/components/app/app-link";
import type { CurriculumModule } from "@/lib/curriculum";
import { routes } from "@/lib/routes";
import { ModuleSwatch } from "./module-swatch";
import { SceneBanner } from "./scene-banner";

/** How far along the module is, in one line: the core as a count solved, the further problems as a size. */
export function moduleProgress({ core, further, coreSolved, solved }: CurriculumModule): string {
  const furtherSolved = solved - coreSolved;
  const furtherPart =
    further.length === 0
      ? null
      : furtherSolved > 0
        ? `${furtherSolved} of ${further.length} further solved`
        : `${further.length} further`;
  const corePart =
    core.length === 0
      ? null
      : coreSolved === core.length
        ? `All ${core.length} core solved`
        : `${coreSolved} of ${core.length} core solved`;
  return [corePart, furtherPart].filter(Boolean).join(", ") || "No problems yet";
}

/**
 * The top of a module's page: its scene as the banner, then its swatch and
 * number with the way back to the area, its title, what it covers, and one
 * line of progress. A module outside every area has no number.
 */
export function ModuleHeader({ module, areaTitle }: { module: CurriculumModule; areaTitle: string }) {
  return (
    <header className="pb-10">
      {module.scene ? <SceneBanner src={module.scene} className="mb-6" /> : null}
      <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className="inline-flex items-center gap-2">
          <ModuleSwatch color={module.color} />
          {module.loose ? "Module" : `Module ${module.numeral}`}
        </span>
        <AppLink to={routes.area(module.areaId)} tone="quiet" className="text-muted-foreground">
          {areaTitle}
        </AppLink>
      </p>
      <h1 className="selectable mt-2 max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
        {module.track.title}
      </h1>
      {module.track.description ? (
        <p className="selectable mt-3 max-w-measure font-serif text-md text-pretty text-muted-foreground">
          {module.track.description}
        </p>
      ) : null}
      <p className="tabular mt-3 text-sm text-muted-foreground">{moduleProgress(module)}</p>
    </header>
  );
}
