import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { useMemo } from "react";
import { AppLink } from "@/components/app/app-link";
import { Page, PageHeader, Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { AreaEntry, areaCounts } from "@/components/curriculum/area-entry";
import { AreaSummary } from "@/components/curriculum/area-summary";
import { ModuleHeader } from "@/components/curriculum/module-header";
import { ModuleNeighbors } from "@/components/curriculum/module-neighbors";
import { ModulePlates } from "@/components/curriculum/module-plates";
import { ModuleRowList } from "@/components/curriculum/module-row-list";
import { ProblemRowList } from "@/components/problem/problem-row-list";
import { useCurriculum } from "@/hooks/use-curriculum";
import { useGo } from "@/hooks/use-go";
import { useProblems } from "@/lib/api/queries";
import {
  areaOrder,
  type Curriculum,
  type CurriculumArea,
  type CurriculumModule,
  findArea,
  findModule,
  moduleEntries,
  nextCore,
  nextInArea,
  siblings,
} from "@/lib/curriculum";
import { moduleColorVars } from "@/lib/module-colors";
import { unmetOutOfOrder, unmetPrerequisites } from "@/lib/prerequisites";
import { routes } from "@/lib/routes";

const areaApi = getRouteApi("/curriculum/$area");
const moduleApi = getRouteApi("/curriculum/$area/$module");

/** The states every Curriculum screen shares: failed, loading, and an empty catalog. */
function CurriculumGate({ children }: { children: (curriculum: Curriculum) => React.ReactNode }) {
  const { curriculum, error, retry } = useCurriculum();
  if (error) {
    return (
      <Page>
        <ErrorState title="The curriculum could not be loaded" error={error} onRetry={retry} />
      </Page>
    );
  }
  if (!curriculum) {
    return (
      <Page>
        <LoadingLines lines={10} className="max-w-measure" />
      </Page>
    );
  }
  if (curriculum.areas.length === 0) {
    return (
      <Page>
        <EmptyState title="There are no modules yet">
          A module is a track under problems/ in the content root. The curriculum fills in once the catalog
          has one.
        </EmptyState>
      </Page>
    );
  }
  return children(curriculum);
}

function NotFound({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Page>
      <EmptyState title={title} action={<AppLink to={routes.curriculum()}>Open the Curriculum</AppLink>}>
        {children}
      </EmptyState>
    </Page>
  );
}

/** An area's screen: its title, where the learner stands, and its modules as plates. */
function AreaView({ area, loose }: { area: CurriculumArea; loose: CurriculumModule[] }) {
  const go = useGo();
  const next = nextCore(area);
  return (
    <Page key={area.id}>
      <div style={moduleColorVars(area.modules)}>
        <PageHeader title={area.title} description={area.description || undefined} />
        <div className="pb-10">
          <AreaSummary area={area} next={next} />
        </div>
        <Section title="Modules" aside={areaCounts(area)}>
          <ModulePlates
            modules={area.modules}
            currentId={next?.module.track.id}
            onOpen={(module) => go(routes.module(area.id, module.track.id))}
          />
        </Section>
      </div>
      {loose.length > 0 ? (
        <Section title="More modules" aside="Outside every area.">
          <ModuleRowList modules={loose} label="Modules outside every area" />
        </Section>
      ) : null}
    </Page>
  );
}

function CurriculumScreen() {
  return (
    <CurriculumGate>
      {(curriculum) => {
        const [only] = curriculum.areas;
        // With one area there is nothing to choose: the Curriculum is that area.
        if (only && curriculum.areas.length === 1) return <AreaView area={only} loose={curriculum.loose} />;
        return (
          <Page>
            <h1 className="sr-only">Curriculum</h1>
            {curriculum.areas.map((area) => (
              <AreaEntry key={area.id} area={area} />
            ))}
          </Page>
        );
      }}
    </CurriculumGate>
  );
}

function AreaScreen() {
  const { area: areaId } = areaApi.useParams();
  return (
    <CurriculumGate>
      {(curriculum) => {
        const area = findArea(curriculum, areaId);
        if (!area) {
          return (
            <NotFound title="There is no such area">Nothing in the curriculum is called "{areaId}".</NotFound>
          );
        }
        return <AreaView area={area} loose={area === curriculum.areas[0] ? curriculum.loose : []} />;
      }}
    </CurriculumGate>
  );
}

/** A module's page: its header, its core and further problems, and the modules beside it. */
function ModuleView({
  curriculum,
  area,
  module,
}: {
  curriculum: Curriculum;
  area: CurriculumArea;
  module: CurriculumModule;
}) {
  const problems = useProblems();
  const around = siblings(curriculum, module);
  const order = useMemo(
    () => (module.loose ? moduleEntries(module).map((entry) => entry.problem) : areaOrder(area)),
    [module, area],
  );
  // Computed over the whole catalog, then cut down to what the page's own order does not already say.
  const unmet = useMemo(
    () => unmetOutOfOrder(unmetPrerequisites(problems.data ?? []), order),
    [problems.data, order],
  );
  const nextId = module.loose ? null : nextInArea(area)?.problem.id;
  const named = module.loose ? module.track.title : `Module ${module.numeral}`;
  return (
    <Page key={module.track.id}>
      <div style={moduleColorVars(around)}>
        <ModuleHeader module={module} areaTitle={area.title} />
        {module.core.length > 0 ? (
          <Section title="Core" aside="The spine of the module, in order.">
            <ProblemRowList
              entries={module.core}
              nextId={nextId}
              unmet={unmet}
              label={`${named} core problems`}
            />
          </Section>
        ) : null}
        {module.further.length > 0 ? (
          <Section
            title="Further"
            aside={
              module.core.length > 0
                ? "More depth on the same ground, outside the recommended route."
                : "This module has no core: all of it is outside the recommended route."
            }
          >
            <ProblemRowList
              entries={module.further}
              nextId={nextId}
              unmet={unmet}
              label={`${named} further problems`}
            />
          </Section>
        ) : null}
        {module.total === 0 ? (
          <EmptyState title="This module has no problems yet" className="pb-10">
            Its problems appear here once the catalog has them.
          </EmptyState>
        ) : null}
        <ModuleNeighbors previous={around[module.number - 2]} next={around[module.number]} />
      </div>
    </Page>
  );
}

function ModuleScreen() {
  const { area: areaId, module: moduleId } = moduleApi.useParams();
  return (
    <CurriculumGate>
      {(curriculum) => {
        const area = findArea(curriculum, areaId);
        if (!area) {
          return (
            <NotFound title="There is no such area">Nothing in the curriculum is called "{areaId}".</NotFound>
          );
        }
        const module = findModule(curriculum, areaId, moduleId);
        if (!module) {
          return (
            <NotFound title="There is no such module">
              {area.title} has no module called "{moduleId}".
            </NotFound>
          );
        }
        return <ModuleView curriculum={curriculum} area={area} module={module} />;
      }}
    </CurriculumGate>
  );
}

export const Route = createLazyRoute("/curriculum")({ component: CurriculumScreen });
export const AreaRoute = createLazyRoute("/curriculum/$area")({ component: AreaScreen });
export const ModuleRoute = createLazyRoute("/curriculum/$area/$module")({ component: ModuleScreen });
