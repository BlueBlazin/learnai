import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { useEffect, useMemo, useRef } from "react";
import { Page, PageHeader } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { PathStage, stageSectionId } from "@/components/path/path-stage";
import { PathSummary } from "@/components/path/path-summary";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useGo } from "@/hooks/use-go";
import { usePaths, useProblems } from "@/lib/api/queries";
import { nextOnPath, pathEntries, unmetOffPath, unmetPrerequisites } from "@/lib/paths";
import { routes } from "@/lib/routes";
import { stageColorVars } from "@/lib/stage-colors";

const route = getRouteApi("/paths");

function PathsScreen() {
  const search = route.useSearch();
  const go = useGo();
  const paths = usePaths();
  const problems = useProblems();

  const path = paths.data?.find((candidate) => candidate.id === search.path) ?? paths.data?.[0];
  const entries = useMemo(
    () => (path && problems.data ? pathEntries(path, problems.data) : []),
    [path, problems.data],
  );
  const next = useMemo(() => nextOnPath(entries), [entries]);
  const unmet = useMemo(
    () => unmetOffPath(unmetPrerequisites(problems.data ?? []), entries),
    [problems.data, entries],
  );

  // Scroll to the stage the address names, once per arrival: not again on later renders.
  const part = search.part;
  const ready = Boolean(path && problems.data);
  const scrolledTo = useRef<string | null>(null);
  useEffect(() => {
    if (!ready || !path || !part) {
      scrolledTo.current = null;
      return;
    }
    const key = `${path.id}:${part}`;
    if (scrolledTo.current === key) return;
    const section = document.getElementById(stageSectionId(part));
    if (!section) return;
    scrolledTo.current = key;
    section.scrollIntoView({ block: "start" });
  }, [ready, path, part]);

  const error = paths.error ?? problems.error;
  if (error) {
    return (
      <Page>
        <ErrorState
          title="The path could not be loaded"
          error={error}
          onRetry={() => {
            void paths.refetch();
            void problems.refetch();
          }}
        />
      </Page>
    );
  }
  if (!paths.data || !problems.data) {
    return (
      <Page>
        <LoadingLines lines={10} className="max-w-measure" />
      </Page>
    );
  }
  if (!path) {
    return (
      <Page>
        <EmptyState title="There is no study path yet">
          A path is a file under paths/ in the content root that lists problems in the order to study them.
          Every problem is in the library meanwhile.
        </EmptyState>
      </Page>
    );
  }

  return (
    <Page>
      <div style={stageColorVars(path)}>
        {paths.data.length > 1 ? (
          <Tabs value={path.id} onValueChange={(id) => go(routes.paths(String(id)), { replace: true })}>
            <TabsList aria-label="Study paths" className="h-toolbar border-b">
              {paths.data.map((candidate) => (
                <TabsTrigger key={candidate.id} value={candidate.id}>
                  {candidate.title}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
        ) : null}
        <div className={paths.data.length > 1 ? "pt-10" : undefined}>
          <PageHeader title={path.title} description={path.description} />
          <PathSummary path={path} entries={entries} next={next} />
          {path.stages.map((stage, index) => (
            <PathStage
              key={stage.title}
              number={index + 1}
              stage={stage}
              entries={entries[index] ?? []}
              nextId={next?.problem.id}
              unmet={unmet}
            />
          ))}
        </div>
      </div>
    </Page>
  );
}

export const Route = createLazyRoute("/paths")({ component: PathsScreen });
