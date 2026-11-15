import { RuledList } from "@/components/app/ruled-list";
import { VerdictMark } from "@/components/results/verdict-mark";
import { Button } from "@/components/ui/button";
import type { Submission } from "@/lib/api";
import { formatDuration, formatRelativeTime } from "@/lib/format";
import { FRAMEWORK_LABEL } from "@/lib/labels";
import { hostOf } from "@/lib/targets";

/**
 * Stored submissions, newest first: verdict, cases passed (and skipped, when
 * any were), framework, time taken, and when; a submission judged on a remote
 * machine names it. With `titles`, each row also names its problem (Home).
 */
export function SubmissionList({
  submissions,
  titles,
  onOpen,
  className,
}: {
  submissions: Submission[];
  /** Problem titles by id; pass to show which problem each submission belongs to. */
  titles?: Map<string, string>;
  onOpen: (submission: Submission) => void;
  className?: string;
}) {
  return (
    <RuledList className={className}>
      {submissions.map((submission) => (
        <li key={submission.id}>
          <Button
            variant="ghost"
            className="h-auto w-full justify-start gap-4 rounded-sm px-2 py-2 text-left font-normal"
            onClick={() => onOpen(submission)}
          >
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              {titles ? (
                <span
                  className="truncate font-serif text-md text-foreground"
                  title={titles.get(submission.problemId) ?? submission.problemId}
                >
                  {titles.get(submission.problemId) ?? submission.problemId}
                </span>
              ) : null}
              <span className="flex items-center gap-3 text-xs">
                <VerdictMark verdict={submission.verdict} />
                <span className="tabular text-muted-foreground">
                  {submission.passed} / {submission.total}
                  {submission.skipped > 0 ? `, ${submission.skipped} skipped` : null}
                </span>
                {hostOf(submission.target) ? (
                  <span className="truncate text-muted-foreground">on {hostOf(submission.target)}</span>
                ) : null}
                {titles ? (
                  <span className="text-muted-foreground">{FRAMEWORK_LABEL[submission.framework]}</span>
                ) : null}
              </span>
            </span>
            {titles ? null : (
              <>
                <span className="w-16 shrink-0 text-xs text-muted-foreground">
                  {FRAMEWORK_LABEL[submission.framework]}
                </span>
                <span className="tabular w-16 shrink-0 text-right text-xs text-muted-foreground">
                  {formatDuration(submission.durationMs)}
                </span>
              </>
            )}
            <span className="w-20 shrink-0 text-right text-xs text-muted-foreground">
              {formatRelativeTime(submission.createdAt)}
            </span>
          </Button>
        </li>
      ))}
    </RuledList>
  );
}
