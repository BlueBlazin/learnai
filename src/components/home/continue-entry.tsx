import { ArrowRight } from "lucide-react";
import { AppLink } from "@/components/app/app-link";
import { StageSwatch } from "@/components/path/stage-swatch";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import type { Path, ProblemSummary, Submission, Track } from "@/lib/api";
import { formatRelativeTime } from "@/lib/format";
import { VERDICT_LABEL } from "@/lib/labels";
import { romanNumeral } from "@/lib/paths";
import { routes } from "@/lib/routes";
import { stageMark } from "@/lib/stage-colors";

/**
 * The problem to pick up next, as the opening of the home page: where you left
 * off if there is recent work, otherwise the next unsolved problem on the study
 * path (or in the catalog, when there is no path). For someone who has not
 * started (`firstRun`) it is the invitation to begin.
 */
export function ContinueEntry({
  problem,
  track,
  lastSubmission,
  onPath,
  stage,
  firstRun = false,
  onOpen,
}: {
  problem: ProblemSummary;
  track?: Track;
  /** The study path the problem is on and its place there; shown instead of the track. */
  onPath?: { path: Path; position: number };
  /**
   * The problem's stage on the path, for the line above the title: its swatch (colored by
   * `stageColorVars` on an ancestor), part number and name, and the place its scene shows.
   */
  stage?: { index: number; title: string; place?: string };
  /** The most recent submission on this problem, when it is the reason for the suggestion. */
  lastSubmission?: Submission;
  /** Nothing has been attempted yet. */
  firstRun?: boolean;
  onOpen: () => void;
}) {
  return (
    <section aria-label="Continue" className="pb-12">
      <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className="text-muted-foreground">
          {lastSubmission ? "Continue where you left off" : firstRun ? "Start here" : "Up next"}
        </span>
        {stage ? (
          <>
            <span className="inline-flex items-center gap-2">
              <StageSwatch color={stageMark(stage.index)} />
              Part {romanNumeral(stage.index + 1)}, {stage.title}
            </span>
            {stage.place ? <span className="ml-auto text-muted-foreground">{stage.place}</span> : null}
          </>
        ) : null}
      </p>
      <h1 className="selectable mt-2 max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
        <AppLink to={routes.problem(problem.id)} tone="quiet">
          {problem.title}
        </AppLink>
      </h1>
      <p className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm text-muted-foreground">
        {onPath ? (
          <span>
            <AppLink to={routes.paths(onPath.path.id)} tone="quiet" className="text-muted-foreground">
              {onPath.path.title}
            </AppLink>
            , No. {onPath.position} of {onPath.path.problemCount}
          </span>
        ) : (
          <span>
            {track?.title ?? problem.track}, No. {problem.order}
          </span>
        )}
        <DifficultyMark difficulty={problem.difficulty} />
        {lastSubmission ? (
          <span>
            Last submitted {formatRelativeTime(lastSubmission.createdAt)}:{" "}
            {VERDICT_LABEL[lastSubmission.verdict].toLowerCase()}
          </span>
        ) : firstRun ? null : (
          <StatusMark status={problem.status} showLabel />
        )}
      </p>
      <div className="mt-6 flex items-center gap-5">
        <Button onClick={onOpen}>
          {firstRun ? "Open the first problem" : "Open problem"}
          <ArrowRight data-icon="inline-end" />
        </Button>
        {firstRun ? (
          <span className="flex gap-5 text-sm">
            {onPath ? <AppLink to={routes.paths(onPath.path.id)}>See the whole path</AppLink> : null}
            <AppLink to={routes.library()}>Browse the library</AppLink>
          </span>
        ) : null}
      </div>
    </section>
  );
}
