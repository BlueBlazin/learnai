import { useMemo } from "react";
import { Page, Section } from "@/components/app/page";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { ContinueEntry } from "@/components/home/continue-entry";
import { ProgressSummary } from "@/components/home/progress-summary";
import { TrackProgress } from "@/components/home/track-progress";
import { SubmissionList } from "@/components/problem/submission-list";
import { useGo } from "@/hooks/use-go";
import { useProblems, useStats, useSubmissions, useTracks } from "@/lib/api/queries";
import { pickContinue } from "@/lib/continue";
import { routes } from "@/lib/routes";

const RECENT_COUNT = 8;

export function HomeScreen() {
  const go = useGo();
  const stats = useStats();
  const tracks = useTracks();
  const problems = useProblems();
  const submissions = useSubmissions();

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
  const error = stats.error ?? tracks.error ?? problems.error ?? submissions.error;

  if (error) {
    return (
      <Page>
        <ErrorState
          title="Your progress could not be loaded"
          error={error}
          onRetry={() => void problems.refetch()}
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
          onOpen={() => go(routes.problem(target.problem.id))}
        />
      )}

      <div className="grid grid-cols-2 gap-x-14">
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
                go(routes.problem(submission.problemId, { tab: "submissions", submission: submission.id }))
              }
            />
          )}
        </Section>
      </div>
    </Page>
  );
}
