import { cn } from "cn";
import type { ProblemSummary } from "@/lib/api";

/**
 * A quiet remark beside a problem's title in a list: it builds on something
 * that is not solved yet. Words, in secondary ink, with no mark and no lock;
 * the row opens exactly as any other. Renders nothing when `unmet` is empty.
 */
export function PrerequisiteNote({ unmet, className }: { unmet: ProblemSummary[]; className?: string }) {
  const [first] = unmet;
  if (!first) return null;
  const titles = unmet.map((problem) => problem.title).join(", ");
  return (
    <span
      title={`Builds on: ${titles} (not solved yet)`}
      className={cn("font-sans text-xs whitespace-nowrap text-muted-foreground", className)}
    >
      best after {unmet.length === 1 ? first.title : `${unmet.length} earlier problems`}
    </span>
  );
}
