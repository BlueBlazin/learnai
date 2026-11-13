import { useMemo } from "react";
import { Page, Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { ContinueEntry } from "@/components/home/continue-entry";
import { GettingStarted } from "@/components/home/getting-started";
import { ProgressSummary } from "@/components/home/progress-summary";
import { TrackProgress } from "@/components/home/track-progress";
import { SubmissionList } from "@/components/problem/submission-list";
import { EnvironmentNotice } from "@/components/workbench/environment-notice";
import { useGo } from "@/hooks/use-go";
import { useEnvStatus, useProblems, useStats, useSubmissions, useTracks } from "@/lib/api/queries";
import { useEnvInstall } from "@/lib/api/use-env-install";
import { pickContinue } from "@/lib/continue";
import { environmentProblem } from "@/lib/environment";
import { routes } from "@/lib/routes";

const RECENT_COUNT = 8;

export function HomeScreen() {
  const go = useGo();
  const stats = useStats();
  const tracks = useTracks();
  const problems = useProblems();
  const submissions = useSubmissions();
  const env = useEnvStatus();
  const { state: install, install: startInstall } = useEnvInstall();

  const titles = useMemo(
    () => new Map((problems.data ?? []).map((problem) => [problem.id, problem.title])),
    [problems.data],
  );
  const target = useMemo(
    () =>
      problems.data && tracks.data && submissions.data
        ? pickContinue(problems.data, tracks.data, submissions.data)
        : undefined,
    [problems.data, tracks.data, submissions.data],
  );
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

  return (
    <Page>
      {target === undefined ? (
        <LoadingLines lines={3} className="pb-12" />
      ) : target === null ? (
        <EmptyState title="Every problem is solved" className="pb-12">
          Nothing is left to pick up. Revisit a track in the library to solve it again in another framework.
        </EmptyState>
      ) : (
        <ContinueEntry
          problem={target.problem}
          track={tracks.data?.find((track) => track.id === target.problem.track)}
          lastSubmission={target.lastSubmission}
          firstRun={firstRun}
          onOpen={() => go(routes.problem(target.problem.id))}
        />
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
            <div>
              <Section title="Progress">
                {stats.data ? <ProgressSummary stats={stats.data} /> : <LoadingLines />}
              </Section>
              <Section title="Tracks">
                {tracks.data ? <TrackProgress tracks={tracks.data} /> : <LoadingLines />}
              </Section>
            </div>
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
                      routes.problem(submission.problemId, { tab: "submissions", submission: submission.id }),
                    )
                  }
                />
              )}
            </Section>
          </>
        )}
      </div>
    </Page>
  );
}
