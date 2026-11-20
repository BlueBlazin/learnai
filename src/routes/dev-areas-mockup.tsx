import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { useMemo } from "react";
import { Page, PageHeader, Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { AreaEntry } from "@/components/mockup/area-entry";
import { AreaSummary } from "@/components/mockup/area-summary";
import { MockupSwitcher } from "@/components/mockup/mockup-switcher";
import {
  buildArea,
  DEFAULT_MODULE,
  type MockupView,
  moduleColorVars,
  moduleEntries,
  PLANNED_AREAS,
  unmetOutOfOrder,
} from "@/components/mockup/model";
import { ModuleHeader } from "@/components/mockup/module-header";
import { ModuleNeighbors } from "@/components/mockup/module-neighbors";
import { ModulePlates } from "@/components/mockup/module-plates";
import { PlannedAreaList } from "@/components/mockup/planned-area-list";
import { PathEntryList } from "@/components/path/path-entry-list";
import { useGo } from "@/hooks/use-go";
import { usePaths, useProblems, useTracks } from "@/lib/api/queries";
import { pluralize } from "@/lib/format";
import { unmetPrerequisites } from "@/lib/paths";
import { routes } from "@/lib/routes";

const route = getRouteApi("/dev/areas-mockup");

/**
 * Development only: a mockup of the catalog organized as areas, modules, and
 * core and further problems. It derives the structure from the real tracks,
 * problems, and first study path, and writes nothing.
 */
function DevAreasMockupScreen() {
  const search = route.useSearch();
  const go = useGo();
  const tracks = useTracks();
  const problems = useProblems();
  const paths = usePaths();

  const view: MockupView = search.view ?? "areas";
  const area = useMemo(
    () =>
      tracks.data && problems.data && paths.data
        ? buildArea(tracks.data, problems.data, paths.data[0] ?? null)
        : null,
    [tracks.data, problems.data, paths.data],
  );
  const unmet = useMemo(
    () =>
      area && problems.data ? unmetOutOfOrder(unmetPrerequisites(problems.data), area.modules) : undefined,
    [area, problems.data],
  );

  const error = tracks.error ?? problems.error ?? paths.error;
  if (error) {
    return (
      <Page>
        <ErrorState
          title="The catalog could not be loaded"
          error={error}
          onRetry={() => {
            for (const query of [tracks, problems, paths]) void query.refetch();
          }}
        />
      </Page>
    );
  }
  if (!area) {
    return (
      <Page>
        <LoadingLines lines={10} className="max-w-measure" />
      </Page>
    );
  }

  const modules = area.modules;
  const module =
    modules.find((candidate) => candidate.track.id === search.module) ??
    modules.find((candidate) => candidate.track.id === DEFAULT_MODULE) ??
    modules[0];
  const entries = module ? moduleEntries(module) : null;

  return (
    // Keyed by what is shown, so each view and each module opens at its top.
    <Page key={view === "module" ? `module:${module?.track.id}` : view}>
      <div style={moduleColorVars(modules)}>
        <MockupSwitcher
          view={view}
          onChange={(next) => go(routes.devAreasMockup(next === "areas" ? undefined : next, search.module))}
          note={`Built from your real catalog; nothing is saved. "${area.title}" is a working title.`}
        />

        {modules.length === 0 ? (
          <EmptyState title="There are no tracks to show">
            The mockup builds its modules from the catalog's tracks, and the catalog has none.
          </EmptyState>
        ) : view === "areas" ? (
          <>
            <h1 className="sr-only">Areas</h1>
            <AreaEntry area={area} />
            <Section title="More areas" aside="Each will have modules and a core of its own.">
              <PlannedAreaList areas={PLANNED_AREAS} />
            </Section>
          </>
        ) : view === "area" ? (
          <>
            <PageHeader title={area.title} description={area.description} />
            <AreaSummary area={area} />
            <Section
              title="Modules"
              aside={`${pluralize(modules.length, "module")}, ${pluralize(area.problemCount, "problem")}, ${area.coreCount} core`}
            >
              <ModulePlates
                modules={modules}
                currentId={area.next?.module.track.id}
                onOpen={(opened) => go(routes.devAreasMockup("module", opened.track.id))}
              />
            </Section>
          </>
        ) : module && entries ? (
          <>
            <ModuleHeader module={module} areaTitle={area.title} />
            {entries.core.length > 0 ? (
              <Section title="Core" aside="The spine of the module, in order.">
                <PathEntryList
                  entries={entries.core}
                  nextId={area.next?.problem.id}
                  unmet={unmet}
                  label={`Module ${module.numeral} core problems`}
                />
              </Section>
            ) : null}
            {entries.further.length > 0 ? (
              <Section
                title="Further"
                aside={
                  entries.core.length > 0
                    ? "More depth on the same ground, outside the recommended route."
                    : "This module has no core: all of it is outside the recommended route."
                }
              >
                <PathEntryList
                  entries={entries.further}
                  unmet={unmet}
                  label={`Module ${module.numeral} further problems`}
                />
              </Section>
            ) : null}
            <ModuleNeighbors previous={modules[module.index - 1]} next={modules[module.index + 1]} />
          </>
        ) : null}
      </div>
    </Page>
  );
}

export const Route = createLazyRoute("/dev/areas-mockup")({ component: DevAreasMockupScreen });
