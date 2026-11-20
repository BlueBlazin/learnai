import { useMemo } from "react";
import { Page, Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { ActivityCalendar } from "@/components/home/activity-calendar";
import { ContinueEntry } from "@/components/home/continue-entry";
import { GettingStarted } from "@/components/home/getting-started";
import { HabitStrip } from "@/components/home/habit-strip";
import { ProgressSummary } from "@/components/home/progress-summary";
import { RouteSummary } from "@/components/home/route-summary";
import { SceneBanner } from "@/components/home/scene-banner";
import { TrackProgress } from "@/components/home/track-progress";
import { StagePlates } from "@/components/path/stage-plates";
import { StageProgressBar } from "@/components/path/stage-progress-bar";
import { SubmissionList } from "@/components/problem/submission-list";
import { ReviewList } from "@/components/review/review-list";
import { EnvironmentNotice } from "@/components/workbench/environment-notice";
import { useGo } from "@/hooks/use-go";
import { activityByDay, solvedThisWeek, weekDays } from "@/lib/activity";
import {
  useEnvStatus,
  usePaths,
  useProblems,
  useReviews,
  useSettings,
  useStats,
  useSubmissions,
  useTracks,
} from "@/lib/api/queries";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { pickContinue } from "@/lib/continue";
import { environmentProblem } from "@/lib/environment";
import { pluralize } from "@/lib/format";
import { findOnPath, nextOnPath, pathEntries, stagesFinished } from "@/lib/paths";
import { dueReviews, reviewEntries, upcomingSummary } from "@/lib/reviews";
import { routes } from "@/lib/routes";
import { sceneUrl } from "@/lib/scenes";
import { stageColorVars } from "@/lib/stage-colors";
import { usePreferences } from "@/stores/preferences";

const RECENT_COUNT = 8;

export function HomeScreen() {
  const go = useGo();
  const stats = useStats();
  const tracks = useTracks();
  const problems = useProblems();
  const submissions = useSubmissions();
  const paths = usePaths();
  const env = useEnvStatus();
  const { state: install, install: startInstall } = useEnvInstall();
  // Reviews never block Home: while they load, fail, or are turned off, nothing about them shows.
  const reviews = useReviews();
  const settings = useSettings();
  const weeklyGoal = usePreferences((state) => state.weeklyGoal);
  const due = useMemo(
    () => reviewEntries(dueReviews(reviews.data ?? []), problems.data ?? [], tracks.data ?? []),
    [reviews.data, problems.data, tracks.data],
  );

  const titles = useMemo(
    () => new Map((problems.data ?? []).map((problem) => [problem.id, problem.title])),
    [problems.data],
  );
  // The study path is the first one the content root defines. A path that fails to
  // load never blocks Home: the suggestion falls back to catalog order.
  const path = paths.data?.[0] ?? null;
  const pathsSettled = paths.data !== undefined || paths.isError;
  const target = useMemo(
    () =>
      problems.data && tracks.data && submissions.data && pathsSettled
        ? pickContinue(problems.data, tracks.data, submissions.data, path)
        : undefined,
    [problems.data, tracks.data, submissions.data, pathsSettled, path],
  );
  const entries = useMemo(
    () => (path && problems.data ? pathEntries(path, problems.data) : null),
    [path, problems.data],
  );
  const nextStep = useMemo(() => (entries ? nextOnPath(entries) : null), [entries]);
  const byDay = useMemo(() => activityByDay(submissions.data ?? []), [submissions.data]);
  const now = new Date();
  // The stage the suggestion sits in, when it is on the path: its scene heads the page.
  const targetStage = useMemo(() => {
    const index = target?.onPath && entries ? findOnPath(entries, target.problem.id)?.stage : undefined;
    const stage = index === undefined ? undefined : path?.stages[index];
    return index === undefined || !stage
      ? undefined
      : { index, title: stage.title, scene: sceneUrl(stage.scene) };
  }, [target, entries, path]);
  // The banner for a finished path is the last stage's.
  const lastScene = nextStep || !path ? null : sceneUrl(path.stages[path.stages.length - 1]?.scene);
  // The page always opens on the place the learner has reached: the suggestion's stage
  // when it is on the path, otherwise the stage the path is at.
  const bannerScene =
    targetStage?.scene ?? (nextStep ? sceneUrl(path?.stages[nextStep.stage]?.scene) : lastScene);
  // Reviews never block Home and can be switched off: while unknown or off, their column is left out.
  const reviewsFact =
    settings.data?.reviewsEnabled && reviews.data
      ? {
          due: due.length,
          firstDueProblemId: due[0]?.review.problemId ?? null,
          upcoming: due.length === 0 ? upcomingSummary(reviews.data, now) : null,
        }
      : null;
  // Until something has been attempted there is no progress to report: Home shows
  // how a problem works and what the tracks cover instead of a page of zeros.
  // Undefined while loading, so the page never shows one layout and then the other.
  const firstRun =
    problems.data && submissions.data
      ? submissions.data.length === 0 && problems.data.every((problem) => problem.status === "todo")
      : undefined;
  const setup = environmentProblem(env.data, null);
  const error = stats.error ?? tracks.error ?? problems.error ?? submissions.error;

  if (error) {
    return (
      <Page>
        <ErrorState
          title="Your progress could not be loaded"
          error={error}
          onRetry={() => {
            for (const query of [stats, tracks, problems, submissions]) void query.refetch();
          }}
        />
      </Page>
    );
  }

  const finished = entries ? stagesFinished(entries) : 0;
  const stageCount = path?.stages.length ?? 0;

  return (
    <Page>
      <div style={path ? stageColorVars(path) : undefined}>
        {target === undefined ? (
          <LoadingLines lines={3} className="pb-12" />
        ) : target === null ? (
          <>
            {lastScene ? <SceneBanner src={lastScene} className="mb-6" /> : null}
            <EmptyState title="Every problem is solved" className="pb-12">
              Nothing is left to pick up. Revisit a track in the library to solve it again in another
              framework.
            </EmptyState>
          </>
        ) : (
          <>
            {bannerScene ? <SceneBanner src={bannerScene} className="mb-6" /> : null}
            <ContinueEntry
              problem={target.problem}
              track={tracks.data?.find((track) => track.id === target.problem.track)}
              lastSubmission={target.lastSubmission}
              onPath={target.onPath}
              stage={targetStage}
              firstRun={firstRun}
              onOpen={() => go(routes.problem(target.problem.id))}
            />
          </>
        )}

        {setup ? (
          <div className="pb-8">
            <EnvironmentNotice
              layout="inline"
              problem={setup}
              install={install}
              onInstall={(framework) => void startInstall(framework)}
            />
          </div>
        ) : null}

        {firstRun === false && stats.data ? (
          <HabitStrip
            streakDays={stats.data.streakDays}
            week={weekDays(byDay, now).days}
            solvedThisWeek={solvedThisWeek(submissions.data ?? [], now)}
            goal={weeklyGoal}
            reviews={reviewsFact}
          />
        ) : null}

        {due.length > 0 ? (
          <Section title="Due for review" aside="Solve each one again from the blank starter.">
            <ReviewList
              entries={due}
              onOpen={(entry) => go(routes.problem(entry.review.problemId, { review: true }))}
            />
          </Section>
        ) : null}

        {path && entries ? (
          <Section
            title="The route"
            aside={
              finished === stageCount
                ? `All ${stageCount} stages finished`
                : finished > 0
                  ? `${finished} of ${stageCount} stages finished`
                  : path.solvedCount > 0
                    ? `The first of ${stageCount} stages is under way`
                    : `${pluralize(stageCount, "stage")}, ${pluralize(path.problemCount, "problem")}`
            }
          >
            <div className="flex flex-col gap-8">
              <div className="flex flex-col gap-4">
                <RouteSummary path={path} />
                <StageProgressBar title={path.title} entries={entries} />
              </div>
              <StagePlates
                path={path}
                entries={entries}
                onOpen={(number) => go(routes.paths(path.id, number))}
              />
            </div>
          </Section>
        ) : null}

        <div className="grid grid-cols-2 gap-x-14">
          {firstRun === undefined ? (
            <>
              <LoadingLines lines={6} />
              <LoadingLines lines={6} />
            </>
          ) : firstRun ? (
            <>
              <Section title="How a problem works">
                <GettingStarted />
              </Section>
              <Section title="What the tracks cover">
                {tracks.data ? <TrackProgress tracks={tracks.data} variant="contents" /> : <LoadingLines />}
              </Section>
            </>
          ) : (
            <>
              <Section title="Progress">
                {stats.data ? (
                  <div className="flex flex-col gap-8">
                    <ActivityCalendar byDay={byDay} now={now} />
                    <ProgressSummary stats={stats.data} />
                  </div>
                ) : (
                  <LoadingLines />
                )}
              </Section>
              <Section title="Recent submissions">
                {!submissions.data ? (
                  <LoadingLines />
                ) : submissions.data.length === 0 ? (
                  <EmptyState title="No submissions yet">
                    Your submitted attempts appear here, newest first, with their verdicts.
                  </EmptyState>
                ) : (
                  <SubmissionList
                    submissions={submissions.data.slice(0, RECENT_COUNT)}
                    titles={titles}
                    onOpen={(submission) =>
                      go(
                        routes.problem(submission.problemId, {
                          tab: "submissions",
                          submission: submission.id,
                        }),
                      )
                    }
                  />
                )}
              </Section>
            </>
          )}
        </div>
      </div>
    </Page>
  );
}
