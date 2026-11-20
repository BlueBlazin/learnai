import { cn } from "cn";
import { AppLink } from "@/components/app/app-link";
import { RuledList } from "@/components/app/ruled-list";
import type { ProblemSummary } from "@/lib/api";
import { routes } from "@/lib/routes";
import { DifficultyMark } from "./difficulty-mark";
import { PrerequisiteNote } from "./prerequisite-note";
import { StatusMark } from "./status-mark";

/**
 * Problems in a reading order, as lines of a table of contents: status, the
 * number, the title (which wraps rather than truncates), and the difficulty.
 * The whole line opens the problem. The suggested next problem is marked with
 * the words "Up next" and nothing else.
 */
export function ProblemRowList({
  entries,
  nextId,
  unmet,
  label,
}: {
  /** Each problem with the number its line carries. */
  entries: { problem: ProblemSummary; position: number }[];
  /** The id of the problem to study next, if it is in this list. */
  nextId?: string | null;
  /** Unsolved prerequisites by problem id (`unmetPrerequisites`). */
  unmet?: Map<string, ProblemSummary[]>;
  /** Accessible name, for example "Module III core problems". */
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
                // The base button does not shrink; a long title has to, so that it wraps.
                className={cn("min-w-0 shrink text-left", isNext && "font-medium")}
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
