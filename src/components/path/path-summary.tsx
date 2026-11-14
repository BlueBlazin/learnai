import { AppLink } from "@/components/app/app-link";
import { ProgressRule } from "@/components/problem/progress-rule";
import type { Path } from "@/lib/api";
import { pluralize } from "@/lib/format";
import { type PathEntry, romanNumeral } from "@/lib/paths";
import { routes } from "@/lib/routes";

/**
 * Where the learner stands on a path, as a sentence: how much is solved and
 * which problem comes next. Before anything is solved it says how long the
 * path is and where it starts, instead of reporting zero.
 */
export function PathSummary({ path, next }: { path: Path; next: PathEntry | null }) {
  const parts = pluralize(path.stages.length, "part");
  const link = next ? <AppLink to={routes.problem(next.problem.id)}>{next.problem.title}</AppLink> : null;
  return (
    <div className="flex max-w-measure flex-col gap-4 pb-10">
      <p className="selectable font-serif text-lg text-pretty">
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
      {path.solvedCount > 0 ? (
        <ProgressRule
          value={path.solvedCount}
          total={path.problemCount}
          label={`${path.title} progress`}
          showCount={false}
        />
      ) : null}
    </div>
  );
}
