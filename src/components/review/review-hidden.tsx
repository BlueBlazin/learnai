import { EyeOff } from "lucide-react";
import type { ReactNode } from "react";
import { EmptyState } from "@/components/app/states";

/**
 * Stands in for whatever a review must not show: hints, the editorial and
 * reference, the code of earlier submissions. Says what is hidden and how to
 * get it back.
 */
export function ReviewHidden({
  what,
  children,
  className,
}: {
  /** What is hidden, as the subject of a sentence: "The hints are". */
  what: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <EmptyState
      className={className ?? "py-0"}
      icon={<EyeOff className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />}
      title="Hidden during a review"
    >
      {children ?? (
        <>
          {what} out of sight until the review is over. If you are stuck, choose "I forgot" above the editor:
          it shows everything again and restarts the schedule.
        </>
      )}
    </EmptyState>
  );
}
