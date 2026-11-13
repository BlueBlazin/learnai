import { useCallback, useEffect, useMemo } from "react";
import { useDefaultLayout } from "react-resizable-panels";
import { AppLink } from "@/components/app/app-link";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { ResultsPanel } from "@/components/results/results-panel";
import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { EditorPane } from "@/components/workbench/editor-pane";
import { StatementPane } from "@/components/workbench/statement-pane";
import { useGo } from "@/hooks/use-go";
import { useRunShortcuts } from "@/hooks/use-run-shortcuts";
import type { RunMode } from "@/lib/api";
import { useEnvStatus, useProblem, useProblems, useTracks } from "@/lib/api/queries";
import { useRun } from "@/lib/api/use-run";
import { useSolutionEditor } from "@/lib/api/use-solution-editor";
import { neighbors } from "@/lib/continue";
import { type ProblemSearch, routes, type StatementTab } from "@/lib/routes";
import { preferredFramework, usePreferences } from "@/stores/preferences";
import { useWorkspaceActions } from "@/stores/workspace-actions";

export function ProblemScreen({ problemId, search }: { problemId: string; search: ProblemSearch }) {
  const go = useGo();
  const problem = useProblem(problemId);
  const problems = useProblems();
  const tracks = useTracks();
  const env = useEnvStatus();

  const savedFramework = usePreferences((state) => state.frameworkByProblem[problemId]);
  const lastFramework = usePreferences((state) => state.lastFramework);
  const setFramework = usePreferences((state) => state.setFramework);
  const framework = problem.data
    ? preferredFramework(problem.data.frameworks, savedFramework, lastFramework)
    : null;

  const editor = useSolutionEditor(problemId, framework);
  const run = useRun(problemId);
  const columns = useDefaultLayout({ id: "whetstone.workspace.columns" });
  const rows = useDefaultLayout({ id: "whetstone.workspace.rows" });

  const { flush } = editor;
  const { start: startRun, active } = run;
  const start = useCallback(
    async (mode: RunMode) => {
      if (!framework || active) return;
      // The judge reads the file on disk: save before running.
      await flush();
      await startRun(framework, mode);
    },
    [active, flush, framework, startRun],
  );
  const onRun = useCallback(() => void start("run"), [start]);
  const onSubmit = useCallback(() => void start("submit"), [start]);
  useRunShortcuts(onRun, onSubmit);

  // Offer Run and Submit in the command palette while this workspace is open.
  const title = problem.data?.title;
  const setActions = useWorkspaceActions((state) => state.setActions);
  useEffect(() => {
    if (!title) return;
    setActions({ problemTitle: title, run: onRun, submit: onSubmit, busy: active });
    return () => setActions(null);
  }, [title, onRun, onSubmit, active, setActions]);

  const tab: StatementTab = search.tab ?? "statement";
  const setSearch = (next: ProblemSearch) => go(routes.problem(problemId, next), { replace: true });

  const around = useMemo(
    () => neighbors(problems.data ?? [], tracks.data ?? [], problemId),
    [problems.data, tracks.data, problemId],
  );
  const prerequisites = useMemo(
    () =>
      (problem.data?.prerequisites ?? []).flatMap(
        (id) => problems.data?.find((candidate) => candidate.id === id) ?? [],
      ),
    [problem.data?.prerequisites, problems.data],
  );

  if (problem.error) {
    return (
      <div className="h-full bg-paper px-gutter pt-10">
        <ErrorState
          title="This problem could not be opened"
          error={problem.error}
          onRetry={() => void problem.refetch()}
        />
        <AppLink to={routes.library()}>Back to the library</AppLink>
      </div>
    );
  }
  if (!problem.data) {
    return (
      <div className="h-full bg-paper px-gutter pt-10">
        <LoadingLines lines={8} className="max-w-measure" />
      </div>
    );
  }
  if (!framework) {
    return (
      <div className="h-full bg-paper px-gutter pt-10">
        <EmptyState title="This problem lists no frameworks">
          Its problem.toml has an empty frameworks list, so there is nothing to write a solution in.
        </EmptyState>
      </div>
    );
  }

  const detail = problem.data;
  return (
    <ResizablePanelGroup
      orientation="horizontal"
      defaultLayout={columns.defaultLayout}
      onLayoutChanged={columns.onLayoutChanged}
    >
      <ResizablePanel id="statement" defaultSize="46%" minSize="30%">
        <StatementPane
          problem={detail}
          track={tracks.data?.find((track) => track.id === detail.track)}
          prerequisites={prerequisites}
          framework={framework}
          tab={tab}
          submissionId={tab === "submissions" ? (search.submission ?? null) : null}
          onTabChange={(next) => setSearch(next === "statement" ? {} : { tab: next })}
          onOpenSubmission={(id) =>
            setSearch(id === null ? { tab: "submissions" } : { tab: "submissions", submission: id })
          }
          previous={around.previous}
          next={around.next}
          onNavigate={(target) => go(routes.problem(target.id))}
        />
      </ResizablePanel>
      <ResizableHandle />
      <ResizablePanel id="work" defaultSize="54%" minSize="34%">
        <ResizablePanelGroup
          orientation="vertical"
          defaultLayout={rows.defaultLayout}
          onLayoutChanged={rows.onLayoutChanged}
        >
          <ResizablePanel id="editor" defaultSize="60%" minSize="25%">
            <EditorPane
              frameworks={detail.frameworks}
              framework={framework}
              solvedFrameworks={detail.solvedFrameworks}
              env={env.data?.frameworks}
              editor={editor}
              busy={active}
              onFrameworkChange={(next) => setFramework(problemId, next)}
              onRun={onRun}
              onSubmit={onSubmit}
            />
          </ResizablePanel>
          <ResizableHandle />
          <ResizablePanel id="results" defaultSize="40%" minSize="15%">
            <ResultsPanel
              run={run.state}
              onCancel={run.cancel}
              actions={
                run.state.submissionId !== null ? (
                  <Button
                    variant="outline"
                    size="xs"
                    onClick={() =>
                      setSearch({ tab: "submissions", submission: run.state.submissionId ?? undefined })
                    }
                  >
                    View submission
                  </Button>
                ) : null
              }
            />
          </ResizablePanel>
        </ResizablePanelGroup>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
