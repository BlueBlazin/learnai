import type { Path } from "@/lib/api";
import { pluralize } from "@/lib/format";

/**
 * Where the learner stands on the path, as one sentence over the progress bar:
 * how much is solved. Before anything is solved it gives the path's length
 * instead of a zero. Which problem comes next is the headline's to say.
 */
export function RouteSummary({ path }: { path: Path }) {
  return (
    <p className="selectable max-w-measure font-serif text-lg text-pretty">
      {path.solvedCount === 0 ? (
        <>
          {pluralize(path.problemCount, "problem")} in {pluralize(path.stages.length, "part")}.
        </>
      ) : path.solvedCount >= path.problemCount ? (
        <>
          All {path.problemCount} problems solved.{" "}
          <span className="text-muted-foreground">The path is finished; the library has more.</span>
        </>
      ) : (
        <>
          <span className="font-semibold tabular">{path.solvedCount}</span> of {path.problemCount} problems
          solved.
        </>
      )}
    </p>
  );
}
