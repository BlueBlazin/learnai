import { ArrowRight } from "lucide-react";
import { AppLink } from "@/components/app/app-link";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import type { ProblemSummary, Submission, Track } from "@/lib/api";
import { formatRelativeTime } from "@/lib/format";
import { VERDICT_LABEL } from "@/lib/labels";
import { routes } from "@/lib/routes";

/**
 * The problem to pick up next, as the opening of the home page: where you left
 * off if there is recent work, otherwise the first unsolved problem.
 */
export function ContinueEntry({
  problem,
  track,
  lastSubmission,
  onOpen,
}: {
  problem: ProblemSummary;
  track?: Track;
  /** The most recent submission on this problem, when it is the reason for the suggestion. */
  lastSubmission?: Submission;
  onOpen: () => void;
}) {
  return (
    <section aria-label="Continue" className="pb-12">
      <p className="text-sm text-muted-foreground">
        {lastSubmission ? "Continue where you left off" : "Up next"}
      </p>
      <h1 className="selectable mt-2 max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
        <AppLink to={routes.problem(problem.id)} tone="quiet">
          {problem.title}
        </AppLink>
      </h1>
      <p className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm text-muted-foreground">
        <span>
          {track?.title ?? problem.track}, No. {problem.order}
        </span>
        <DifficultyMark difficulty={problem.difficulty} />
        {lastSubmission ? (
          <span>
            Last submitted {formatRelativeTime(lastSubmission.createdAt)}:{" "}
            {VERDICT_LABEL[lastSubmission.verdict].toLowerCase()}
          </span>
        ) : (
          <StatusMark status={problem.status} showLabel />
        )}
      </p>
      <Button className="mt-6" onClick={onOpen}>
        Open problem
        <ArrowRight data-icon="inline-end" />
      </Button>
    </section>
  );
}
