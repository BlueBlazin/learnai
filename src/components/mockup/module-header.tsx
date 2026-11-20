import { AppLink } from "@/components/app/app-link";
import { SceneBanner } from "@/components/home/scene-banner";
import { StageSwatch } from "@/components/path/stage-swatch";
import { routes } from "@/lib/routes";
import { stageMark } from "@/lib/stage-colors";
import type { Module } from "./model";

/** How far along the module is, in one line: the core as a count solved, the further problems as a size. */
function progressLine({ core, further, coreSolved, furtherSolved }: Module): string {
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
 * line of progress. The colors come from `moduleColorVars` on an ancestor.
 */
export function ModuleHeader({ module, areaTitle }: { module: Module; areaTitle: string }) {
  return (
    <header className="pb-10">
      {module.scene ? <SceneBanner src={module.scene} className="mb-6" /> : null}
      <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className="inline-flex items-center gap-2">
          <StageSwatch color={stageMark(module.index)} />
          Module {module.numeral}
        </span>
        <AppLink to={routes.devAreasMockup("area")} tone="quiet" className="text-muted-foreground">
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
      <p className="tabular mt-3 text-sm text-muted-foreground">{progressLine(module)}</p>
    </header>
  );
}
