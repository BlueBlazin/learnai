import { AppLink } from "@/components/app/app-link";
import { RuledList } from "@/components/app/ruled-list";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { PrerequisiteNote } from "@/components/problem/prerequisite-note";
import { StatusMark } from "@/components/problem/status-mark";
import type { ProblemSummary } from "@/lib/api";
import type { PathEntry } from "@/lib/paths";
import { routes } from "@/lib/routes";

/**
 * The problems of one stage as lines of a table of contents: status, the
 * running number along the path, the title (which wraps rather than truncates),
 * and the difficulty. The whole line opens the problem. The suggested next
 * problem is marked with the words "Up next" and nothing else.
 */
export function PathEntryList({
  entries,
  nextId,
  unmet,
  label,
}: {
  entries: PathEntry[];
  /** The id of the next unsolved problem on the path, if it is in this list. */
  nextId?: string | null;
  /** Unsolved prerequisites by problem id (`unmetPrerequisites`). */
  unmet?: Map<string, ProblemSummary[]>;
  /** Accessible name, for example "Part I problems". */
  label: string;
}) {
  return (
    <RuledList ordered framed={false} className="min-w-0 flex-1" aria-label={label}>
      {entries.map(({ problem, position }) => {
        const isNext = problem.id === nextId;
        return (
          <li
            key={problem.id}
            value={position}
            aria-current={isNext ? "step" : undefined}
            className="relative flex items-baseline gap-3 px-2 py-2 transition-colors hover:bg-accent/60"
          >
            <StatusMark status={problem.status} className="self-center" />
            <span className="tabular w-6 shrink-0 text-right text-xs text-faint" aria-hidden>
              {position}
            </span>
            <span className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-3 font-serif text-md text-pretty">
              <AppLink
                to={routes.problem(problem.id)}
                tone="quiet"
                stretched
                className={isNext ? "font-medium" : undefined}
              >
                {problem.title}
              </AppLink>
              <PrerequisiteNote unmet={unmet?.get(problem.id) ?? []} />
            </span>
            {isNext ? <span className="shrink-0 text-xs font-medium whitespace-nowrap">Up next</span> : null}
            <DifficultyMark difficulty={problem.difficulty} className="w-20 shrink-0 self-center text-xs" />
          </li>
        );
      })}
    </RuledList>
  );
}
