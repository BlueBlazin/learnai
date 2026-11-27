import { memo, useRef } from "react";
import { AppLink } from "@/components/app/app-link";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { useGo } from "@/hooks/use-go";
import { useWindowedRows } from "@/hooks/use-windowed-rows";
import type { ProblemSummary } from "@/lib/api";
import { routes } from "@/lib/routes";
import { DifficultyMark } from "./difficulty-mark";
import { FrameworkList } from "./framework-list";
import { JudgeList } from "./judge-list";
import { PrerequisiteNote } from "./prerequisite-note";
import { StatusMark } from "./status-mark";

/** Row height in pixels before a rendered row has been measured. */
const ROW_HEIGHT = 38.5;
const COLUMNS = 6;

const NONE: ProblemSummary[] = [];

const ProblemRow = memo(function ProblemRow({
  problem,
  index,
  unmet,
}: {
  problem: ProblemSummary;
  index: number;
  /** Prerequisites of this problem that are not solved yet. */
  unmet: ProblemSummary[];
}) {
  const go = useGo();
  return (
    <TableRow
      data-windowed-row
      aria-rowindex={index + 1}
      className="cursor-pointer"
      onClick={() => go(routes.problem(problem.id))}
    >
      <TableCell className="w-8 pr-0">
        <StatusMark status={problem.status} />
      </TableCell>
      <TableCell className="tabular w-10 text-right text-xs text-faint">{problem.order}</TableCell>
      <TableCell className="max-w-0 font-serif text-md">
        <span className="flex items-baseline gap-3">
          <AppLink tone="quiet" to={routes.problem(problem.id)} truncate className="min-w-0">
            {problem.title}
          </AppLink>
          <PrerequisiteNote unmet={unmet} className="min-w-0 shrink-0" />
        </span>
      </TableCell>
      <TableCell className="w-24 text-xs">
        <DifficultyMark difficulty={problem.difficulty} level={problem.level} />
      </TableCell>
      <TableCell className="w-40 truncate text-xs text-muted-foreground">
        <JudgeList judges={problem.judges} className="block" />
      </TableCell>
      <TableCell className="w-40 text-xs">
        <FrameworkList frameworks={problem.frameworks} solved={problem.solvedFrameworks} />
      </TableCell>
    </TableRow>
  );
});

/** Stands in for the rows of a windowed table that are not rendered. */
function SpacerRow({ height }: { height: number }) {
  if (height <= 0) return null;
  return (
    <TableRow aria-hidden className="border-0 hover:bg-transparent">
      <TableCell colSpan={COLUMNS} className="p-0" style={{ height }} />
    </TableRow>
  );
}

/**
 * Problems as rows: status, number, title, difficulty, judges, frameworks. Column
 * widths are fixed so consecutive tables (one per track) align as one. Only the
 * rows near the viewport are rendered (a table that is off screen keeps a single
 * row and its height), so a catalog of hundreds of problems scrolls and filters
 * as fast as a short one.
 *
 * `unmet` (from `unmetPrerequisites`) adds a quiet "best after ..." beside the
 * title of a problem whose prerequisites are not solved yet. Nothing is locked.
 */
export function ProblemTable({
  problems,
  label,
  unmet,
}: {
  problems: ProblemSummary[];
  label: string;
  unmet?: Map<string, ProblemSummary[]>;
}) {
  const body = useRef<HTMLTableSectionElement>(null);
  const { start, end, before, after } = useWindowedRows(body, problems.length, {
    estimatedRowHeight: ROW_HEIGHT,
  });
  return (
    <Table aria-label={label} aria-rowcount={problems.length} className="table-fixed">
      {/* Fixed layout sizes columns from the first row, which may be a spacer: state them here. */}
      <colgroup>
        <col className="w-8" />
        <col className="w-10" />
        <col />
        <col className="w-24" />
        <col className="w-40" />
        <col className="w-40" />
      </colgroup>
      <TableBody ref={body}>
        <SpacerRow height={before} />
        {problems.slice(start, end).map((problem, offset) => (
          <ProblemRow
            key={problem.id}
            problem={problem}
            index={start + offset}
            unmet={unmet?.get(problem.id) ?? NONE}
          />
        ))}
        <SpacerRow height={after} />
      </TableBody>
    </Table>
  );
}
