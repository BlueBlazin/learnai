import { cn } from "cn";
import type { ReactNode } from "react";
import type { RunMode, Verdict } from "@/lib/api";
import { describeEnv, summarizeRun } from "@/lib/run-summary";
import { VerdictMark } from "./verdict-mark";

/**
 * The outcome of a finished run: the verdict in display type, one sentence
 * saying what happened, and the environment it ran in.
 */
export function VerdictHeader({
  verdict,
  mode,
  passed,
  total,
  durationMs,
  env = {},
  actions,
  className,
}: {
  verdict: Verdict;
  mode: RunMode | null;
  passed: number;
  total: number;
  durationMs: number;
  env?: Record<string, string>;
  actions?: ReactNode;
  className?: string;
}) {
  const environment = describeEnv(env);
  return (
    <div className={cn("flex items-start justify-between gap-4", className)} role="status">
      <div className="min-w-0">
        <VerdictMark verdict={verdict} size="lg" className="font-serif text-lg" />
        <p className="selectable mt-0.5 text-sm text-muted-foreground">
          {summarizeRun({ verdict, mode, passed, total, durationMs })}
        </p>
        {environment ? <p className="selectable mt-1 text-xs text-faint">{environment}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
