import { cn } from "cn";
import { DisclosureChevron } from "@/components/app/disclosure-chevron";
import { RuledList } from "@/components/app/ruled-list";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { CaseInfo, CaseResult } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import { CaseMark } from "./case-mark";
import { MetricList } from "./metric-list";
import { OutputBlock } from "./output-block";

type CaseDisplayState = "pending" | "running" | "done" | "not-run";

function hasDetail(result: CaseResult | null): result is CaseResult {
  return (
    result !== null &&
    (result.message !== null || result.detail !== null || result.stdout !== "" || result.metrics.length > 0)
  );
}

function Summary({
  info,
  state,
  result,
  expandable,
}: {
  info: CaseInfo;
  state: CaseDisplayState;
  result: CaseResult | null;
  expandable: boolean;
}) {
  const mark = state === "done" && result ? result.status : state === "done" ? "not-run" : state;
  return (
    <span className="flex h-8 w-full items-center gap-2.5 px-3 text-left text-sm">
      <DisclosureChevron className={expandable ? undefined : "invisible"} />
      <CaseMark status={mark} />
      <span
        title={info.name}
        className={cn(
          "min-w-0 truncate",
          state === "pending" || state === "not-run" ? "text-muted-foreground" : "text-foreground",
        )}
      >
        {info.name}
      </span>
      <span className="min-w-0 flex-1 truncate text-xs text-muted-foreground group-data-panel-open/disclosure:invisible">
        {result?.message ?? (state === "not-run" ? "Not run" : null)}
      </span>
      {result && result.status !== "skipped" ? (
        <span className="tabular shrink-0 text-xs text-muted-foreground">
          {formatDuration(result.durationMs)}
        </span>
      ) : null}
    </span>
  );
}

/**
 * One test case in a run or a stored submission. Shows state, name, the one-line
 * failure reason, and time; opens to the full message, detail, captured output,
 * and metrics when there is anything to show.
 */
function CaseRow({
  info,
  state,
  result,
  defaultOpen = false,
}: {
  info: CaseInfo;
  state: CaseDisplayState;
  result: CaseResult | null;
  defaultOpen?: boolean;
}) {
  const failed = result?.status === "fail" || result?.status === "error" || result?.status === "timeout";
  if (!hasDetail(result)) {
    return (
      <li>
        <Summary info={info} state={state} result={result} expandable={false} />
      </li>
    );
  }
  return (
    <li>
      <Collapsible defaultOpen={defaultOpen}>
        <CollapsibleTrigger className="group/disclosure block w-full rounded-sm outline-none hover:bg-accent/60 focus-visible:bg-accent">
          <Summary info={info} state={state} result={result} expandable />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="flex flex-col gap-3 pt-1 pr-3 pb-4 pl-12">
            {result.message ? (
              <p className={cn("selectable text-sm", failed ? "text-fail" : "text-foreground")}>
                {result.message}
              </p>
            ) : null}
            <MetricList metrics={result.metrics} />
            {result.detail ? <OutputBlock label="Detail">{result.detail}</OutputBlock> : null}
            {result.stdout ? <OutputBlock label="Printed output">{result.stdout}</OutputBlock> : null}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </li>
  );
}

/** The list of cases for a live run or a stored submission. The first failure starts open. */
export function CaseList({
  cases,
  ended,
}: {
  cases: { info: CaseInfo; state: "pending" | "running" | "done"; result: CaseResult | null }[];
  /** True once the run is over: cases that never ran are labeled as such. */
  ended: boolean;
}) {
  const firstFailure = cases.findIndex(
    (c) => c.result && c.result.status !== "pass" && c.result.status !== "skipped",
  );
  return (
    <RuledList ordered>
      {cases.map((c, index) => (
        <CaseRow
          key={c.info.id}
          info={c.info}
          state={ended && c.state !== "done" ? "not-run" : c.state}
          result={c.result}
          defaultOpen={index === firstFailure}
        />
      ))}
    </RuledList>
  );
}
