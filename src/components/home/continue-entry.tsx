import { ArrowRight } from "lucide-react";
import { AppLink } from "@/components/app/app-link";
import { ModuleSwatch } from "@/components/curriculum/module-swatch";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { StatusMark } from "@/components/problem/status-mark";
import { Button } from "@/components/ui/button";
import type { ProblemSummary, Submission, Track } from "@/lib/api";
import type { Place } from "@/lib/curriculum";
import { formatRelativeTime } from "@/lib/format";
import { VERDICT_LABEL } from "@/lib/labels";
import { routes } from "@/lib/routes";

/**
 * The problem to pick up next, as the opening of the home page: where you left
 * off if there is recent work, otherwise the next unsolved core problem of the
 * first area. It names the problem's module, with its swatch, and for a core
 * problem its place on the area's route. For someone who has not started
 * (`firstRun`) it is the invitation to begin.
 */
export function ContinueEntry({
  problem,
  track,
  lastSubmission,
  place,
  firstRun = false,
  onOpen,
}: {
  problem: ProblemSummary;
  /** Named instead of the module while the curriculum does not place the problem. */
  track?: Track;
  /** Where the problem sits in the curriculum: its module, its area, and its place on the route. */
  place?: Place | null;
  /** The most recent submission on this problem, when it is the reason for the suggestion. */
  lastSubmission?: Submission;
  /** Nothing has been attempted yet. */
  firstRun?: boolean;
  onOpen: () => void;
}) {
  const module = place?.module;
  const area = place?.area;
  return (
    <section aria-label="Continue" className="pb-12">
      <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className="text-muted-foreground">
          {lastSubmission ? "Continue where you left off" : firstRun ? "Start here" : "Up next"}
        </span>
        {module ? (
          <span className="inline-flex items-center gap-2">
            <ModuleSwatch color={module.color} />
            <AppLink to={routes.module(module.areaId, module.track.id)} tone="quiet">
              {module.loose ? module.track.title : `Module ${module.numeral}, ${module.track.title}`}
            </AppLink>
          </span>
        ) : null}
      </p>
      <h1 className="selectable mt-2 max-w-measure font-serif text-3xl font-medium tracking-tight text-balance">
        <AppLink to={routes.problem(problem.id)} tone="quiet">
          {problem.title}
        </AppLink>
      </h1>
      <p className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 text-sm text-muted-foreground">
        {area ? (
          <span>
            <AppLink to={routes.area(area.id)} tone="quiet" className="text-muted-foreground">
              {area.title}
            </AppLink>
            {place.routePosition === null
              ? ", further"
              : `, core ${place.routePosition} of ${area.coreCount}`}
          </span>
        ) : module ? null : (
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
            <AppLink to={routes.curriculum()}>See the curriculum</AppLink>
            <AppLink to={routes.library()}>Browse the library</AppLink>
          </span>
        ) : null}
      </div>
    </section>
  );
}
