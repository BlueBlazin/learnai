import { ArrowLeft } from "lucide-react";
import { MetaItem, MetaList } from "@/components/app/meta-list";
import { Section } from "@/components/app/page";
import { ErrorState, LoadingLines } from "@/components/app/states";
import { CodeBlock } from "@/components/prose/code-block";
import { CaseList } from "@/components/results/case-list";
import { MetricList } from "@/components/results/metric-list";
import { VerdictHeader } from "@/components/results/verdict-header";
import { ReviewHidden } from "@/components/review/review-hidden";
import { Button } from "@/components/ui/button";
import type { SubmissionDetail } from "@/lib/api";
import { formatDateTime } from "@/lib/format";
import { FRAMEWORK_LABEL, isNotJudged } from "@/lib/labels";
import { notJudgedReason } from "@/lib/run-summary";
import { hostOf } from "@/lib/targets";

/**
 * One stored submission: its verdict, case results, and the code as it was
 * submitted. During a review (`codeHidden`) the code and the case details stay
 * out of sight: they are the answer being recalled.
 */
export function SubmissionView({
  submission,
  error,
  onBack,
  onRetry,
  codeHidden = false,
}: {
  /** A review is in progress: show the outcome, not the code. */
  codeHidden?: boolean;
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
            skipped={submission.skipped}
            durationMs={submission.durationMs}
            target={submission.target}
          />
          {isNotJudged(submission.verdict) ? (
            <p className="selectable text-sm">{notJudgedReason(submission.verdict)}</p>
          ) : null}
          <MetaList ruled>
            <MetaItem label="Submitted">{formatDateTime(submission.createdAt)}</MetaItem>
            <MetaItem label="Framework">{FRAMEWORK_LABEL[submission.framework]}</MetaItem>
            <MetaItem label="Ran on">{hostOf(submission.target) ?? "This computer"}</MetaItem>
            {submission.review ? <MetaItem label="Attempt">Review, from the blank starter</MetaItem> : null}
          </MetaList>
          <MetricList metrics={submission.metrics} />
          {codeHidden ? (
            <ReviewHidden what="The code and case details of earlier submissions are" className="py-2" />
          ) : (
            <>
              <CaseList
                ended
                cases={submission.cases.map(({ name, public: isPublic, stage, ...result }) => ({
                  info: { id: result.id, name, public: isPublic, ...(stage ? { stage } : {}) },
                  state: "done" as const,
                  result,
                }))}
              />
              <Section title="Code as submitted" className="pb-0">
                <CodeBlock code={submission.code} language="python" lineNumbers />
              </Section>
            </>
          )}
        </>
      )}
    </div>
  );
}
