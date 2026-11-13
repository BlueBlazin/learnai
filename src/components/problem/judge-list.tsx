import { cn } from "cn";
import { Fragment } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { JudgeKind } from "@/lib/api";
import { JUDGE_DESCRIPTION, JUDGE_LABEL } from "@/lib/labels";

/**
 * How a problem is judged, as words. With `explain`, each judge has a tooltip
 * describing what it checks (problem header); without, it is plain text (tables).
 */
export function JudgeList({
  judges,
  explain = false,
  className,
}: {
  judges: JudgeKind[];
  explain?: boolean;
  className?: string;
}) {
  if (!explain) {
    return (
      <span className={cn("truncate", className)}>
        {judges.map((judge) => JUDGE_LABEL[judge]).join(", ")}
      </span>
    );
  }
  return (
    <span className={className}>
      {judges.map((judge, index) => (
        <Fragment key={judge}>
          {index > 0 ? ", " : null}
          <Tooltip>
            <TooltipTrigger
              render={<Button variant="link" />}
              className="type-inherit h-auto rounded-sm p-0 text-foreground underline decoration-faint decoration-dotted underline-offset-4"
            >
              {JUDGE_LABEL[judge]}
            </TooltipTrigger>
            <TooltipContent>{JUDGE_DESCRIPTION[judge]}</TooltipContent>
          </Tooltip>
        </Fragment>
      ))}
    </span>
  );
}
