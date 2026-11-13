import type { ReactNode } from "react";
import { SHORTCUTS, ShortcutHint } from "@/components/app/shortcut-hint";
import { EmptyState, ErrorState } from "@/components/app/states";
import { ScrollArea } from "@/components/ui/scroll-area";
import { isActive, logText, type RunState } from "@/lib/api/run-state";
import { CaseList } from "./case-row";
import { MetricList } from "./metric-list";
import { OutputBlock } from "./output-block";
import { RunProgress } from "./run-progress";
import { VerdictHeader } from "./verdict-header";

/**
 * Everything a run reports, live: preparing, streaming case results, the
 * verdict with metrics and limits, failure detail, captured output, and the
 * canceled and judge-failure states.
 */
export function ResultsPanel({
  run,
  onCancel,
  actions,
}: {
  run: RunState;
  onCancel: () => void;
  /** Controls shown beside the verdict, for example a link to the stored submission. */
  actions?: ReactNode;
}) {
  if (run.phase === "idle") {
    return (
      <div className="h-full bg-paper px-4">
        <EmptyState title="No results yet">
          <p>
            Run checks your code against the public cases. Submit runs every case and records the attempt.
          </p>
          <p className="mt-3 flex gap-5">
            <ShortcutHint label="Run" keys={SHORTCUTS.run} />
            <ShortcutHint label="Submit" keys={SHORTCUTS.submit} />
          </p>
        </EmptyState>
      </div>
    );
  }

  const active = isActive(run);
  const finished = run.cases.filter((c) => c.state === "done").length;
  const diagnostics = logText(run.logs);
  // Diagnostics matter when the judge failed and `end` carried no detail of its own.
  const showDiagnostics =
    diagnostics !== "" &&
    run.detail === null &&
    (run.verdict === "harness_error" || run.verdict === "import_error");

  return (
    <ScrollArea className="h-full bg-paper">
      <div className="flex flex-col gap-4 px-4 pt-3.5 pb-8">
        {active ? (
          <RunProgress
            mode={run.mode}
            message={run.phase === "running" ? null : (run.preparingMessage ?? "Starting")}
            finished={finished}
            total={run.total}
            canceling={run.canceling}
            onCancel={onCancel}
          />
        ) : run.startError ? (
          <ErrorState title="The run could not start" error={run.startError} className="py-0" />
        ) : run.verdict ? (
          <VerdictHeader
            verdict={run.verdict}
            mode={run.mode}
            passed={run.passed}
            total={run.total}
            durationMs={run.durationMs}
            env={run.env}
            actions={actions}
          />
        ) : null}

        {run.message ? <p className="selectable text-sm text-fail">{run.message}</p> : null}
        {run.detail ? <OutputBlock label="Detail">{run.detail}</OutputBlock> : null}
        <MetricList metrics={run.metrics} />
        {showDiagnostics ? (
          <OutputBlock label="Judge output" tone={run.verdict === "import_error" ? "fail" : "neutral"}>
            {diagnostics}
          </OutputBlock>
        ) : null}
        {run.cases.length > 0 ? <CaseList cases={run.cases} ended={!active} /> : null}
      </div>
    </ScrollArea>
  );
}
