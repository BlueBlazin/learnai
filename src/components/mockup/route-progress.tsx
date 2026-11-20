import { AppLink } from "@/components/app/app-link";
import { romanNumeral } from "@/lib/paths";
import { routes } from "@/lib/routes";
import {
  type NextProblem,
  PATH_TITLE,
  PROBLEM_COUNT,
  STAGES,
  stageColor,
  stageSolved,
  stagesFinished,
} from "./mock-data";

/**
 * Progress along the path as one bar of a segment per stage. A segment is as
 * wide as its stage has problems; its track is the stage color mixed faintly
 * into the paper, and the solved share is the full color, a whole problem at a
 * time.
 */
export function SegmentedProgress({ solved }: { solved: number }) {
  return (
    <div
      role="progressbar"
      aria-label={`${PATH_TITLE} progress`}
      aria-valuemin={0}
      aria-valuemax={PROBLEM_COUNT}
      aria-valuenow={solved}
      aria-valuetext={`${solved} of ${PROBLEM_COUNT} problems solved, ${stagesFinished(solved)} of ${STAGES.length} stages finished`}
      className="flex h-2.5 gap-0.75"
    >
      {STAGES.map((stage) => {
        const color = stageColor(stage);
        const total = stage.problems.length;
        return (
          <span
            key={stage.number}
            className="h-full min-w-0 basis-0"
            style={{
              flexGrow: total,
              backgroundColor: `light-dark(color-mix(in oklab, ${color} 22%, var(--paper)), color-mix(in oklab, ${color} 30%, var(--paper)))`,
            }}
          >
            <span
              className="block h-full"
              style={{ width: `${(stageSolved(stage, solved) / total) * 100}%`, backgroundColor: color }}
            />
          </span>
        );
      })}
    </div>
  );
}

/**
 * Where the learner stands on the route: the sentence `PathSummary` sets on
 * the Path screen today, over the segmented bar in place of its plain rule.
 */
export function RouteProgress({ solved, next }: { solved: number; next: NextProblem | null }) {
  return (
    <div className="flex flex-col gap-4">
      <p className="selectable max-w-measure font-serif text-lg text-pretty">
        {next ? (
          <>
            <span className="font-semibold tabular">{solved}</span> of {PROBLEM_COUNT} problems solved.{" "}
            <span className="text-muted-foreground">
              Next is <AppLink to={routes.problem(next.id)}>{next.title}</AppLink>, in Part{" "}
              {romanNumeral(next.stage.number)}.
            </span>
          </>
        ) : (
          <>
            All {PROBLEM_COUNT} problems solved.{" "}
            <span className="text-muted-foreground">The path is finished; the library has more.</span>
          </>
        )}
      </p>
      <SegmentedProgress solved={solved} />
    </div>
  );
}
