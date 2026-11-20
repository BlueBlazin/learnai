import { useMemo } from "react";
import { Page, Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { AreaSummary } from "@/components/curriculum/area-summary";
import { ModulePlates } from "@/components/curriculum/module-plates";
import { SceneBanner } from "@/components/curriculum/scene-banner";
import { ActivityCalendar } from "@/components/home/activity-calendar";
import { ContinueEntry } from "@/components/home/continue-entry";
import { GettingStarted } from "@/components/home/getting-started";
import { HabitStrip } from "@/components/home/habit-strip";
import { ProgressSummary } from "@/components/home/progress-summary";
import { TrackProgress } from "@/components/home/track-progress";
import { SubmissionList } from "@/components/problem/submission-list";
import { ReviewList } from "@/components/review/review-list";
import { EnvironmentNotice } from "@/components/workbench/environment-notice";
import { useCurriculum } from "@/hooks/use-curriculum";
import { useGo } from "@/hooks/use-go";
import { activityByDay, solvedThisWeek, weekDays } from "@/lib/activity";
import {
  useEnvStatus,
  useProblems,
  useReviews,
  useSettings,
  useStats,
  useSubmissions,
  useTracks,
} from "@/lib/api/queries";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { pickContinue } from "@/lib/continue";
import { coresComplete, nextCore } from "@/lib/curriculum";
import { environmentProblem } from "@/lib/environment";
import { pluralize } from "@/lib/format";
import { moduleColorVars } from "@/lib/module-colors";
import { dueReviews, reviewEntries, upcomingSummary } from "@/lib/reviews";
import { routes } from "@/lib/routes";
import { usePreferences } from "@/stores/preferences";

const RECENT_COUNT = 8;

export function HomeScreen() {
  const go = useGo();
  const stats = useStats();
  const tracks = useTracks();
  const problems = useProblems();
  const submissions = useSubmissions();
  const { curriculum } = useCurriculum();
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
  // Home follows the first area. The suggestion waits for the curriculum, so it never
  // shows one problem and then another.
  const area = curriculum?.areas[0] ?? null;
  const target = useMemo(
    () =>
      problems.data && tracks.data && submissions.data && curriculum
        ? pickContinue(problems.data, tracks.data, submissions.data, curriculum)
        : undefined,
    [problems.data, tracks.data, submissions.data, curriculum],
  );
  const nextStep = useMemo(() => (area ? nextCore(area) : null), [area]);
  const byDay = useMemo(() => activityByDay(submissions.data ?? []), [submissions.data]);
  const now = new Date();
  // The banner for an area whose core is finished is its last module's.
  const lastScene = area?.modules[area.modules.length - 1]?.scene ?? null;
  // The page always opens on the place the learner has reached: the module that holds
  // the suggestion, otherwise the module the route is at.
  const bannerScene = target?.place?.module.scene ?? nextStep?.module.scene ?? lastScene;
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
  const cores = area ? coresComplete(area.modules) : null;

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

  return (
    <Page>
      <div style={area ? moduleColorVars(area.modules) : undefined}>
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
              place={target.place}
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

        {area && cores ? (
          <Section
            title="The route"
            aside={
              cores.withCore === 0
                ? `${pluralize(area.modules.length, "module")}, ${pluralize(area.problemCount, "problem")}`
                : cores.complete === cores.withCore
                  ? `Core finished in all ${pluralize(cores.withCore, "module")}`
                  : cores.complete > 0
                    ? `Core finished in ${cores.complete} of ${pluralize(cores.withCore, "module")}`
                    : area.coreSolved > 0
                      ? `The first of ${pluralize(area.modules.length, "module")} is under way`
                      : `${pluralize(area.modules.length, "module")}, ${pluralize(area.coreCount, "core problem")}`
            }
          >
            <div className="flex flex-col gap-8">
              <AreaSummary area={area} />
              <ModulePlates
                compact
                modules={area.modules}
                currentId={nextStep?.module.track.id}
                onOpen={(module) => go(routes.module(area.id, module.track.id))}
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
              <Section title="What the modules cover">
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
