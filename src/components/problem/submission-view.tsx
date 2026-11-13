import { ArrowLeft } from "lucide-react";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { CodeBlock } from "@/components/prose/code-block";
import { CaseList } from "@/components/results/case-row";
import { MetricList } from "@/components/results/metric-list";
import { VerdictHeader } from "@/components/results/verdict-header";
import { Button } from "@/components/ui/button";
import type { SubmissionDetail } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import { FRAMEWORK_LABEL } from "@/lib/labels";

/** One stored submission: its verdict, case results, and the code as it was submitted. */
export function SubmissionView({
  submission,
  error,
  onBack,
  onRetry,
}: {
  /** Undefined while loading. */
  submission?: SubmissionDetail;
  error?: unknown;
  onBack: () => void;
  onRetry?: () => void;
}) {
  return (
    <div className="flex max-w-measure flex-col gap-5">
      <Button variant="ghost" size="sm" className="-ml-2 self-start text-muted-foreground" onClick={onBack}>
        <ArrowLeft data-icon="inline-start" />
        All submissions
      </Button>
      {error ? (
        <ErrorState title="The submission could not be loaded" error={error} onRetry={onRetry} />
      ) : !submission ? (
        <LoadingLines lines={6} />
      ) : (
        <>
          <VerdictHeader
            verdict={submission.verdict}
            mode={submission.mode}
            passed={submission.passed}
            total={submission.total}
            durationMs={submission.durationMs}
          />
          <MetaList className="border-y border-border py-2.5">
            <MetaItem label="Submitted">{formatDateTime(submission.createdAt)}</MetaItem>
            <MetaItem label="Framework">{FRAMEWORK_LABEL[submission.framework]}</MetaItem>
            <MetaItem label="Target">{submission.target}</MetaItem>
          </MetaList>
          <MetricList metrics={submission.metrics} />
          <CaseList
            ended
            cases={submission.cases.map((result, index) => ({
              info: { id: result.id, name: `Case ${index + 1}`, public: true },
              state: "done" as const,
              result,
            }))}
          />
          <section className="border-t border-rule pt-3">
            <h2 className="pb-3 text-sm font-semibold">Code as submitted</h2>
            <CodeBlock code={submission.code} language="python" lineNumbers />
          </section>
        </>
      )}
    </div>
  );
}
