import { AppLink } from "@/components/app/app-link";
import { ProgressRule } from "@/components/problem/progress-rule";
import type { Path } from "@/lib/api";
import type { PathEntry } from "@/lib/paths";
import { routes } from "@/lib/routes";

/**
 * The study path on Home: its title (linking to the path), how far along it the
 * learner is, and the next problem on it.
 */
export function PathProgress({ path, next }: { path: Path; next: PathEntry | null }) {
  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex items-center gap-4">
        <span className="w-56 shrink-0 font-serif text-md">
          <AppLink to={routes.paths(path.id)} tone="quiet">
            {path.title}
          </AppLink>
        </span>
        <ProgressRule
          value={path.solvedCount}
          total={path.problemCount}
          label={`${path.title} progress`}
          className="min-w-0 flex-1"
        />
      </div>
      <p className="selectable text-sm text-pretty text-muted-foreground">
        {next ? (
          <>
            Next on the path, No. {next.position}:{" "}
            <AppLink to={routes.problem(next.problem.id)}>{next.problem.title}</AppLink>
          </>
        ) : (
          "Every problem on the path is solved."
        )}
      </p>
    </div>
  );
}
