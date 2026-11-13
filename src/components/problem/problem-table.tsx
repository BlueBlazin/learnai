import { memo } from "react";
import { AppLink } from "@/components/app/app-link";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";
import { useGo } from "@/hooks/use-go";
import type { ProblemSummary } from "@/lib/api";
import { routes } from "@/lib/routes";
import { DifficultyMark } from "./difficulty-mark";
import { FrameworkList } from "./framework-list";
import { JudgeList } from "./judge-list";
import { StatusMark } from "./status-mark";

const ProblemRow = memo(function ProblemRow({ problem }: { problem: ProblemSummary }) {
  const go = useGo();
  return (
    <TableRow className="cursor-pointer" onClick={() => go(routes.problem(problem.id))}>
      <TableCell className="w-8 pr-0">
        <StatusMark status={problem.status} />
      </TableCell>
      <TableCell className="tabular w-10 text-right text-xs text-faint">{problem.order}</TableCell>
      <TableCell className="max-w-0 truncate font-serif text-md">
        <AppLink tone="quiet" to={routes.problem(problem.id)} truncate>
          {problem.title}
        </AppLink>
      </TableCell>
      <TableCell className="w-24 text-xs">
        <DifficultyMark difficulty={problem.difficulty} />
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

/**
 * Problems as rows: status, number, title, difficulty, judges, frameworks. Column
 * widths are fixed so consecutive tables (one per track) align as one.
 */
export function ProblemTable({ problems, label }: { problems: ProblemSummary[]; label: string }) {
  return (
    <Table aria-label={label} className="table-fixed">
      <TableBody>
        {problems.map((problem) => (
          <ProblemRow key={problem.id} problem={problem} />
        ))}
      </TableBody>
    </Table>
  );
}
