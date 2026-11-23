import { createLazyRoute, getRouteApi } from "@tanstack/react-router";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type Layout, type LayoutChangedMeta, useDefaultLayout } from "react-resizable-panels";
import { SkillEarned } from "@/components/accomplishments/skill-earned";
import { AppLink } from "@/components/app/app-link";
import { ScreenMessage } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { AssetNoticeList, type NoticeAsset } from "@/components/models/asset-notice";
import { ResultsPanel } from "@/components/results/results-panel";
import { ReviewDueNotice } from "@/components/review/review-due-notice";
import { ReviewOutcomeNote } from "@/components/review/review-outcome";
import { ReviewStrip } from "@/components/review/review-strip";
import { TargetPicker } from "@/components/targets/target-picker";
import { Button } from "@/components/ui/button";
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { EditorPane } from "@/components/workbench/editor-pane";
import { HardwareNotice } from "@/components/workbench/hardware-notice";
import { PanelToggle } from "@/components/workbench/panel-toggle";
import { StatementPane } from "@/components/workbench/statement-pane";
import { SuggestionsStatus } from "@/components/workbench/suggestions-status";
import { useCollapsiblePanel } from "@/hooks/use-collapsible-panel";
import { useCurriculum } from "@/hooks/use-curriculum";
import { useEarnedSkills } from "@/hooks/use-earned-skills";
import { useGo } from "@/hooks/use-go";
import { usePanelShortcuts } from "@/hooks/use-panel-shortcuts";
import { useRunShortcuts } from "@/hooks/use-run-shortcuts";
import type { Review, ReviewAction, RunMode } from "@/lib/api";
import {
  useAssets,
  useEnvStatus,
  useProblem,
  useProblems,
  useResolveReview,
  useReviews,
  useSubmissions,
  useTargets,
} from "@/lib/api/queries";
import { useAssetFetch } from "@/lib/api/use-asset-fetch";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { useRun } from "@/lib/api/use-run";
import { useSolutionEditor } from "@/lib/api/use-solution-editor";
import { neighbors } from "@/lib/continue";
import { placeOf } from "@/lib/curriculum";
import { editorSession, useEditorSession } from "@/lib/editor";
import { missingHardware } from "@/lib/hardware";
import { IS_MAC } from "@/lib/platform";
import { reviewOutcome } from "@/lib/reviews";
import { type ProblemSearch, routes, type StatementTab } from "@/lib/routes";
import { hostOf, knownHardware, LOCAL, resolveTarget, targetsWith } from "@/lib/targets";
import { isCollapsedSize, withPanelExpanded } from "@/lib/workspace-layout";
import { preferredFramework, usePreferences } from "@/stores/preferences";
import { useWorkspaceActions } from "@/stores/workspace-actions";

const route = getRouteApi("/problems/$problemId");

/** The share (percent) of the workspace each collapsible panel takes until the user resizes it. */
const STATEMENT_SIZE = 46;
const RESULTS_SIZE = 40;

function ProblemScreen({ problemId }: { problemId: string }) {
  const search = route.useSearch();
  const go = useGo();
  const problem = useProblem(problemId);
  const problems = useProblems();
  const { curriculum } = useCurriculum();
  const env = useEnvStatus();
  const { state: install, install: startInstall } = useEnvInstall();

  const savedFramework = usePreferences((state) => state.frameworkByProblem[problemId]);
  const lastFramework = usePreferences((state) => state.lastFramework);
  const setFramework = usePreferences((state) => state.setFramework);
  const markFrameworkUsed = usePreferences((state) => state.markFrameworkUsed);
  const framework = problem.data
    ? preferredFramework(problem.data.frameworks, savedFramework, lastFramework)
    : null;

  // Review mode lives in the address, so it survives a reload. It holds only while
  // the problem has an active review; until the list arrives the address is trusted.
  const reviews = useReviews();
  const review = reviews.data?.find((candidate) => candidate.problemId === problemId) ?? null;
  const reviewKnown = reviews.data !== undefined || reviews.isError;
  const reviewing = search.review === true && (reviewKnown ? review !== null : true);
  const resolveReview = useResolveReview(problemId);
  /** The schedule as it was when the review submit in the results panel started. */
  const [reviewed, setReviewed] = useState<Review | null>(null);

  const editor = useSolutionEditor(problemId, framework, reviewing);
  const suggestions = useEditorSession();
  const run = useRun(problemId);
  // An accepted submit that earns a skill says so, once, after the run has ended.
  const earned = useEarnedSkills(run.state);

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
  const columns = useDefaultLayout({ id: "learnai.workspace.columns" });
  const rows = useDefaultLayout({ id: "learnai.workspace.rows" });

  // The statement and the results collapse to nothing; the editor between them never does.
  // A collapsed panel is a zero in the saved layout, so it stays collapsed across a restart.
  const panelSizes = usePreferences((state) => state.panelSizes);
  const setPanelSize = usePreferences((state) => state.setPanelSize);
  // An address that opens a stored submission asks for the statement side: arrive with it showing.
  const [openingColumns] = useState(() =>
    search.submission === undefined
      ? columns.defaultLayout
      : withPanelExpanded(columns.defaultLayout, "statement", panelSizes.statement ?? STATEMENT_SIZE),
  );
  const editorPanel = useRef<HTMLDivElement | null>(null);
  /** Collapsing a panel that held the focus hands it to the editor rather than dropping it. */
  const focusEditor = useCallback(
    () => editorPanel.current?.querySelector<HTMLElement>("[role=textbox]")?.focus(),
    [],
  );
  const statementPanel = useCollapsiblePanel({
    initiallyCollapsed: isCollapsedSize(openingColumns?.statement),
    restoreSize: panelSizes.statement,
    onFocusLost: focusEditor,
  });
  const resultsPanel = useCollapsiblePanel({
    initiallyCollapsed: isCollapsedSize(rows.defaultLayout?.results),
    restoreSize: panelSizes.results,
    onFocusLost: focusEditor,
  });
  const { expand: showStatement, toggle: toggleStatement } = statementPanel;
  const { expand: showResults, toggle: toggleResults } = resultsPanel;
  usePanelShortcuts(toggleStatement, toggleResults);
  // Save each layout, and remember the size a panel showed at for when it reopens.
  const { onLayoutChanged: saveColumns } = columns;
  const onColumnsChanged = useCallback(
    (layout: Layout, meta: LayoutChangedMeta) => {
      saveColumns(layout, meta);
      setPanelSize("statement", layout.statement);
    },
    [saveColumns, setPanelSize],
  );
  const { onLayoutChanged: saveRows } = rows;
  const onRowsChanged = useCallback(
    (layout: Layout, meta: LayoutChangedMeta) => {
      saveRows(layout, meta);
      setPanelSize("results", layout.results);
    },
    [saveRows, setPanelSize],
  );

  const { flush } = editor;
  const { start: startRun, active } = run;
  const start = useCallback(
    async (mode: RunMode) => {
      if (!framework || active) return;
      // A run reports into the results panel: show it.
      showResults();
      markFrameworkUsed(problemId, framework);
      setReviewed(reviewing && mode === "submit" ? review : null);
      // The judge reads the file on disk: save before running.
      await flush();
      await startRun(framework, mode, target, reviewing);
    },
    [
      active,
      flush,
      framework,
      markFrameworkUsed,
      problemId,
      review,
      reviewing,
      showResults,
      startRun,
      target,
    ],
  );
  const onRun = useCallback(() => void start("run"), [start]);
  const onSubmit = useCallback(() => void start("submit"), [start]);
  useRunShortcuts(onRun, onSubmit);

  // Offer Run, Submit, and the panel toggles in the command palette while this workspace is open.
  const title = problem.data?.title;
  const setActions = useWorkspaceActions((state) => state.setActions);
  const statementCollapsed = statementPanel.collapsed;
  const resultsCollapsed = resultsPanel.collapsed;
  useEffect(() => {
    if (!title) return;
    setActions({
      problemTitle: title,
      run: onRun,
      submit: onSubmit,
      busy: active,
      statementCollapsed,
      resultsCollapsed,
      toggleStatement,
      toggleResults,
    });
    return () => setActions(null);
  }, [
    title,
    onRun,
    onSubmit,
    active,
    statementCollapsed,
    resultsCollapsed,
    toggleStatement,
    toggleResults,
    setActions,
  ]);

  const tab: StatementTab = search.tab ?? "statement";
  /**
   * Moves within the workspace (tab, open submission) without leaving review mode.
   * Where it leads is on the statement side, so the statement is shown.
   */
  const setSearch = (next: ProblemSearch) => {
    showStatement();
    go(routes.problem(problemId, reviewing ? { ...next, review: true } : next), { replace: true });
  };
  const { tab: searchTab, submission: searchSubmission, review: searchReview } = search;
  const setReviewMode = useCallback(
    (on: boolean) => {
      const rest: ProblemSearch = {
        ...(searchTab ? { tab: searchTab } : {}),
        ...(searchSubmission ? { submission: searchSubmission } : {}),
      };
      go(routes.problem(problemId, on ? { ...rest, review: true } : rest), { replace: true });
    },
    [go, problemId, searchSubmission, searchTab],
  );

  // An address that asks for a review the problem does not have (none scheduled,
  // already resolved, reviews turned off) falls back to the normal workspace.
  useEffect(() => {
    if (searchReview && reviewKnown && !review) setReviewMode(false);
  }, [searchReview, reviewKnown, review, setReviewMode]);

  // An accepted review submit completes the review: the attempt is gone from disk,
  // so drop any unsaved edit to it and return to the normal workspace.
  const reviewAccepted =
    run.state.phase === "done" &&
    run.state.review &&
    run.state.mode === "submit" &&
    run.state.verdict === "accepted";
  const { discard } = editor;
  const acceptedRun = reviewAccepted ? run.state.runId : null;
  /** The accepted review run that review mode was already left for: once per run. */
  const leftFor = useRef<string | null>(null);
  useEffect(() => {
    if (acceptedRun === null || leftFor.current === acceptedRun) return;
    leftFor.current = acceptedRun;
    discard();
    setReviewMode(false);
  }, [acceptedRun, discard, setReviewMode]);
  const outcome = reviewAccepted && reviewed ? reviewOutcome(reviewed, reviews.data) : null;

  const endReview = async (action: ReviewAction) => {
    // A snoozed attempt is kept for tomorrow; the other two discard it.
    if (action === "snooze") await flush();
    else discard();
    try {
      await resolveReview.mutateAsync(action);
    } catch {
      // The strip shows the reason; the review goes on.
      return;
    }
    discard();
    setReviewMode(false);
  };

  const around = useMemo(
    () => (curriculum ? neighbors(curriculum, problemId) : { previous: null, next: null }),
    [curriculum, problemId],
  );
  const place = useMemo(() => (curriculum ? placeOf(curriculum, problemId) : null), [curriculum, problemId]);
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
    <>
      <ResizablePanelGroup
        orientation="horizontal"
        defaultLayout={openingColumns}
        onLayoutChanged={onColumnsChanged}
      >
        <ResizablePanel
          id="statement"
          defaultSize={`${STATEMENT_SIZE}%`}
          minSize="30%"
          {...statementPanel.panelProps}
        >
          <StatementPane
            problem={detail}
            place={place}
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
            reviewing={reviewing}
          />
        </ResizablePanel>
        <ResizableHandle />
        <ResizablePanel id="work" defaultSize={`${100 - STATEMENT_SIZE}%`} minSize="34%">
          <ResizablePanelGroup
            orientation="vertical"
            defaultLayout={rows.defaultLayout}
            onLayoutChanged={onRowsChanged}
          >
            <ResizablePanel
              id="editor"
              defaultSize={`${100 - RESULTS_SIZE}%`}
              minSize="25%"
              elementRef={editorPanel}
            >
              <EditorPane
                problemId={problemId}
                layoutControls={
                  <>
                    <PanelToggle
                      panel="statement"
                      collapsed={statementPanel.collapsed}
                      onToggle={toggleStatement}
                    />
                    <PanelToggle
                      panel="results"
                      collapsed={resultsPanel.collapsed}
                      onToggle={toggleResults}
                    />
                  </>
                }
                suggestions={
                  <SuggestionsStatus session={suggestions} onRetry={() => editorSession.restart()} />
                }
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
                review={reviewing}
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
                    {reviewing && review ? (
                      <ReviewStrip
                        review={review}
                        busy={active || resolveReview.isPending}
                        error={resolveReview.error ? resolveReview.error.message : null}
                        onForgot={() => void endReview("forgot")}
                        onSnooze={() => void endReview("snooze")}
                        onRetire={() => void endReview("retire")}
                        onStartOver={() => void editor.reset()}
                        onLeave={() => setReviewMode(false)}
                      />
                    ) : review?.due && !reviewing && !reviewAccepted ? (
                      <ReviewDueNotice
                        review={review}
                        disabled={active}
                        onStart={() => setReviewMode(true)}
                      />
                    ) : null}
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
            <ResizablePanel
              id="results"
              defaultSize={`${RESULTS_SIZE}%`}
              minSize="15%"
              {...resultsPanel.panelProps}
            >
              <ResultsPanel
                run={run.state}
                onCancel={run.cancel}
                idleHint={
                  reviewing
                    ? "Run checks your review attempt against the public cases. Submit runs every case; an accepted submit completes this review."
                    : undefined
                }
                note={outcome ? <ReviewOutcomeNote outcome={outcome} /> : null}
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
      <SkillEarned {...earned} />
    </>
  );
}

/** Keyed by problem so editor, run, and scroll state never leak between problems. */
function ProblemRoute() {
  const { problemId } = route.useParams();
  return <ProblemScreen key={problemId} problemId={problemId} />;
}

export const Route = createLazyRoute("/problems/$problemId")({ component: ProblemRoute });
