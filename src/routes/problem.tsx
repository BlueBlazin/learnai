import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo } from "react";
import { useDefaultLayout } from "react-resizable-panels";
import { AppLink } from "@/components/app/app-link";
import { ScreenMessage } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { AssetNoticeList, type NoticeAsset } from "@/components/models/asset-notice";
import { ResultsPanel } from "@/components/results/results-panel";
import { TargetPicker } from "@/components/targets/target-picker";
import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { EditorPane } from "@/components/workbench/editor-pane";
import { HardwareNotice } from "@/components/workbench/hardware-notice";
import { StatementPane } from "@/components/workbench/statement-pane";
import { useGo } from "@/hooks/use-go";
import { useRunShortcuts } from "@/hooks/use-run-shortcuts";
import type { RunMode } from "@/lib/api";
import {
  useAssets,
  useEnvStatus,
  useProblem,
  useProblems,
  useSubmissions,
  useTargets,
  useTracks,
} from "@/lib/api/queries";
import { useAssetFetch } from "@/lib/api/use-asset-fetch";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { useRun } from "@/lib/api/use-run";
import { useSolutionEditor } from "@/lib/api/use-solution-editor";
import { neighbors } from "@/lib/continue";
import { missingHardware } from "@/lib/hardware";
import { IS_MAC } from "@/lib/platform";
import { type ProblemSearch, routes, type StatementTab } from "@/lib/routes";
import { hostOf, knownHardware, LOCAL, resolveTarget, targetsWith } from "@/lib/targets";
import { preferredFramework, usePreferences } from "@/stores/preferences";
import { useWorkspaceActions } from "@/stores/workspace-actions";

const route = getRouteApi("/problems/$problemId");

function ProblemScreen({ problemId }: { problemId: string }) {
  const search = route.useSearch();
  const go = useGo();
  const problem = useProblem(problemId);
  const problems = useProblems();
  const tracks = useTracks();
  const env = useEnvStatus();
  const { state: install, install: startInstall } = useEnvInstall();

  const savedFramework = usePreferences((state) => state.frameworkByProblem[problemId]);
  const lastFramework = usePreferences((state) => state.lastFramework);
  const setFramework = usePreferences((state) => state.setFramework);
  const markFrameworkUsed = usePreferences((state) => state.markFrameworkUsed);
  const framework = problem.data
    ? preferredFramework(problem.data.frameworks, savedFramework, lastFramework)
    : null;

  const editor = useSolutionEditor(problemId, framework);
  const run = useRun(problemId);

  // Where runs execute: the remembered choice, while that target still exists.
  const targets = useTargets();
  const preferredTarget = usePreferences((state) => state.target);
  const setTarget = usePreferences((state) => state.setTarget);
  const target = resolveTarget(targets.data, preferredTarget);
  const host = hostOf(target);

  // Models the problem declares: listed only when it declares any.
  const declared = problem.data?.assets ?? [];
  const assets = useAssets(declared.length > 0);
  const fetch = useAssetFetch();
  // Downloads for the machine the runs go to: a remote run's missing model is fetched there.
  const targetFetch = useAssetFetch(target);
  const submissions = useSubmissions(problemId);

  // Editing or running counts as working in this framework; it becomes the default elsewhere.
  const edited = editor.saveState !== "saved";
  useEffect(() => {
    if (edited && framework) markFrameworkUsed(problemId, framework);
  }, [edited, framework, markFrameworkUsed, problemId]);
  const columns = useDefaultLayout({ id: "whetstone.workspace.columns" });
  const rows = useDefaultLayout({ id: "whetstone.workspace.rows" });

  const { flush } = editor;
  const { start: startRun, active } = run;
  const start = useCallback(
    async (mode: RunMode) => {
      if (!framework || active) return;
      markFrameworkUsed(problemId, framework);
      // The judge reads the file on disk: save before running.
      await flush();
      await startRun(framework, mode, target);
    },
    [active, flush, framework, markFrameworkUsed, problemId, startRun, target],
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
      <ScreenMessage>
        <ErrorState
          title="This problem could not be opened"
          error={problem.error}
          onRetry={() => void problem.refetch()}
        />
        <AppLink to={routes.library()}>Back to the library</AppLink>
      </ScreenMessage>
    );
  }
  if (!problem.data) {
    return (
      <ScreenMessage>
        <LoadingLines lines={8} className="max-w-measure" />
      </ScreenMessage>
    );
  }
  if (!framework) {
    return (
      <ScreenMessage>
        <EmptyState title="This problem lists no frameworks">
          Its problem.toml has an empty frameworks list, so there is nothing to write a solution in.
        </EmptyState>
      </ScreenMessage>
    );
  }

  const detail = problem.data;
  // The declared models, with their state where the listing knows them.
  const models: NoticeAsset[] = (assets.data?.assets ?? []).filter((asset) => declared.includes(asset.name));
  // A run that skipped cases for a model names it itself, even if the listing has not loaded.
  const skippedFor: NoticeAsset[] = run.state.missingAssets.map(
    (missing) => models.find((model) => model.name === missing.name) ?? missing,
  );
  // Evidence about hardware belongs to one machine: a run or a submission on
  // another target says nothing about the one chosen now.
  const sameTarget = (run.state.target ?? LOCAL) === target;
  const lastSubmission = submissions.data?.[0];
  const lacking = missingHardware(detail.requires, {
    run: sameTarget ? run.state : { requires: null, requiresMet: null },
    lastVerdict: lastSubmission?.target === target ? lastSubmission.verdict : undefined,
    isMac: host === null && IS_MAC,
    known: knownHardware(targets.data?.find((candidate) => candidate.name === target)),
  });
  // A remote machine whose last check found the hardware, to point to.
  const capable = lacking
    ? targetsWith(targets.data ?? [], lacking).find((candidate) => candidate.name !== target)
    : undefined;
  // What a remote run skipped for: its machine's models are not in the local listing.
  const remoteSkippedFor: NoticeAsset[] = run.state.missingAssets.map(({ name, sizeMb }) => ({
    name,
    sizeMb,
  }));
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
              env={env.data}
              install={install}
              onInstall={(target) => void startInstall(target)}
              editor={editor}
              busy={active}
              onFrameworkChange={(next) => setFramework(problemId, next)}
              onRun={onRun}
              onSubmit={onSubmit}
              remote={host !== null}
              targetPicker={
                <TargetPicker
                  targets={targets.data ?? []}
                  value={target}
                  disabled={active}
                  onChange={setTarget}
                />
              }
              notices={
                <>
                  {/* The listing describes this computer's disk, so it speaks for local runs only. */}
                  {host === null ? <AssetNoticeList assets={models} fetch={fetch} /> : null}
                  {lacking ? (
                    <HardwareNotice
                      requires={lacking}
                      machine={host ?? undefined}
                      remote={capable?.label}
                      onUseRemote={capable && !active ? () => setTarget(capable.name) : undefined}
                    />
                  ) : null}
                </>
              }
            />
          </ResizablePanel>
          <ResizableHandle />
          <ResizablePanel id="results" defaultSize="40%" minSize="15%">
            <ResultsPanel
              run={run.state}
              onCancel={run.cancel}
              resolution={
                <AssetNoticeList
                  assets={sameTarget && host ? remoteSkippedFor : skippedFor}
                  fetch={sameTarget ? targetFetch : fetch}
                  host={sameTarget ? (host ?? undefined) : undefined}
                  layout="inline"
                  whenPresent={`${run.state.mode === "run" ? "Run" : "Submit"} again to judge the skipped cases.`}
                />
              }
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

/** Keyed by problem so editor, run, and scroll state never leak between problems. */
function ProblemRoute() {
  const { problemId } = route.useParams();
  return <ProblemScreen key={problemId} problemId={problemId} />;
}

export const Route = createLazyRoute("/problems/$problemId")({ component: ProblemRoute });
