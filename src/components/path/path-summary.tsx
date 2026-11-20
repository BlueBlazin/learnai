import { AppLink } from "@/components/app/app-link";
import { StageProgressBar } from "@/components/path/stage-progress-bar";
import type { Path } from "@/lib/api";
import { pluralize } from "@/lib/format";
import { type PathEntry, romanNumeral } from "@/lib/paths";
import { routes } from "@/lib/routes";

/**
 * Where the learner stands on a path, as a sentence: how much is solved and
 * which problem comes next. Before anything is solved it says how long the
 * path is and where it starts, instead of reporting zero. The stage bar under
 * it is always shown; before anything is solved its tinted tracks introduce
 * the stages' colors. The colors come from `stageColorVars` on an ancestor.
 */
export function PathSummary({
  path,
  entries,
  next,
}: {
  path: Path;
  entries: PathEntry[][];
  next: PathEntry | null;
}) {
  const parts = pluralize(path.stages.length, "part");
  const link = next ? <AppLink to={routes.problem(next.problem.id)}>{next.problem.title}</AppLink> : null;
  return (
    <div className="flex flex-col gap-4 pb-10">
      <p className="selectable max-w-measure font-serif text-lg text-pretty">
        {path.solvedCount === 0 ? (
          <>
            {pluralize(path.problemCount, "problem")} in {parts}.{" "}
            {link ? <span className="text-muted-foreground">It starts with {link}.</span> : null}
          </>
        ) : next ? (
          <>
            <span className="font-semibold tabular">{path.solvedCount}</span> of {path.problemCount} problems
            solved.{" "}
            <span className="text-muted-foreground">
              Next is {link}, in Part {romanNumeral(next.stage + 1)}.
            </span>
          </>
        ) : (
          <>
            All {path.problemCount} problems solved.{" "}
            <span className="text-muted-foreground">The path is finished; the library has more.</span>
          </>
        )}
      </p>
      <StageProgressBar title={path.title} entries={entries} />
    </div>
  );
}
