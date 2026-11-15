import { cn } from "cn";
import type { ReactNode } from "react";
import type { RunMode, Verdict } from "@/lib/api";
import { describeEnv, summarizeRun } from "@/lib/run-summary";
import { hostOf } from "@/lib/targets";
import { VerdictMark } from "./verdict-mark";

/**
 * The outcome of a finished run: the verdict in display type, one sentence
 * saying what happened, and the environment it ran in (and on which machine,
 * when that was not this computer).
 */
export function VerdictHeader({
  verdict,
  mode,
  passed,
  total,
  skipped = 0,
  durationMs,
  env = {},
  target,
  actions,
  className,
}: {
  verdict: Verdict;
  mode: RunMode | null;
  passed: number;
  total: number;
  /** Cases that were not run. */
  skipped?: number;
  durationMs: number;
  env?: Record<string, string>;
  /** The target the run executed on; named in the environment line when it is remote. */
  target?: string | null;
  actions?: ReactNode;
  className?: string;
}) {
  const host = target ? hostOf(target) : null;
  const environment = [host ? `Ran on ${host}` : "", describeEnv(env)].filter(Boolean).join(": ");
  return (
    <div className={cn("flex items-start justify-between gap-4", className)} role="status">
      <div className="min-w-0">
        <VerdictMark verdict={verdict} size="lg" className="font-serif text-lg" />
        <p className="selectable mt-0.5 text-sm text-muted-foreground">
          {summarizeRun({ verdict, mode, passed, total, skipped, durationMs })}
        </p>
        {environment ? <p className="selectable mt-1 text-xs text-faint">{environment}</p> : null}
      </div>
      {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
    </div>
  );
}
