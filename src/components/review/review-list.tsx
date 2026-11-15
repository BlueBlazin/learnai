import { RuledList } from "@/components/app/ruled-list";
import { Button } from "@/components/ui/button";
import type { ReviewEntry } from "@/lib/reviews";
import { overdueLabel, reviewLabel } from "@/lib/reviews";

/**
 * Reviews that are due, longest overdue first: the problem, its track, which
 * review this is, and how late it is. A row opens the workspace in review mode.
 */
export function ReviewList({
  entries,
  onOpen,
  now,
  className,
}: {
  entries: ReviewEntry[];
  onOpen: (entry: ReviewEntry) => void;
  /** The moment "overdue" is counted from; the current time when absent. */
  now?: Date;
  className?: string;
}) {
  return (
    <RuledList className={className} aria-label="Due for review">
      {entries.map((entry) => (
        <li key={entry.review.problemId}>
          <Button
            variant="ghost"
            className="h-auto w-full justify-start gap-4 rounded-sm px-2 py-2 text-left font-normal"
            onClick={() => onOpen(entry)}
          >
            <span className="flex min-w-0 flex-1 flex-col gap-0.5">
              <span className="truncate font-serif text-md text-foreground" title={entry.title}>
                {entry.title}
              </span>
              <span className="flex items-center gap-3 text-xs text-muted-foreground">
                {entry.track ? <span className="truncate">{entry.track}</span> : null}
                <span className="shrink-0">{reviewLabel(entry.review)}</span>
              </span>
            </span>
            <span className="tabular w-28 shrink-0 text-right text-xs text-muted-foreground">
              {overdueLabel(entry.review.dueAt, now)}
            </span>
          </Button>
        </li>
      ))}
    </RuledList>
  );
}
