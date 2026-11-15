import { ChevronLeft, ChevronRight } from "lucide-react";
import { IconButton } from "@/components/app/icon-button";
import { EmptyState, ErrorState, LoadingLines } from "@/components/app/states";
import { EditorialPanel } from "@/components/problem/editorial-panel";
import { HintList } from "@/components/problem/hint-list";
import { ProblemHeader } from "@/components/problem/problem-header";
import { SubmissionList } from "@/components/problem/submission-list";
import { SubmissionView } from "@/components/problem/submission-view";
import { ProblemFigures } from "@/components/prose/problem-figures";
import { Prose } from "@/components/prose/prose";
import { ReviewHidden } from "@/components/review/review-hidden";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { Framework, ProblemDetail, ProblemSummary, Track } from "@/lib/api";
import { useReference, useSubmission, useSubmissions, useUnlockEditorial } from "@/lib/api/queries";
import type { StatementTab } from "@/lib/routes";
import { usePreferences } from "@/stores/preferences";
import { PaneToolbar } from "./pane-toolbar";

/** The scrolling page inside each tab: paper background, generous margins that tighten in a narrow pane. */
function PageScroll({ children }: { children: React.ReactNode }) {
  return (
    <ScrollArea className="h-full bg-paper">
      <div className="@container">
        <div className="px-6 pt-8 pb-24 @xl:px-gutter @xl:pt-10">{children}</div>
      </div>
    </ScrollArea>
  );
}

/**
 * The left half of the workspace: the statement and its companions (hints,
 * editorial, submissions) as tabs over one reading surface. During a review
 * (`reviewing`) the hints, the editorial and reference, and the code of earlier
 * submissions are replaced by a line saying they are hidden.
 */
export function StatementPane({
  problem,
  track,
  prerequisites,
  framework,
  tab,
  submissionId,
  onTabChange,
  onOpenSubmission,
  previous,
  next,
  onNavigate,
  reviewing = false,
}: {
  /** A review of this problem is in progress. */
  reviewing?: boolean;
  problem: ProblemDetail;
  track?: Track;
  prerequisites: ProblemSummary[];
  framework: Framework | null;
  tab: StatementTab;
  /** The submission open in the Submissions tab, if any. */
  submissionId: number | null;
  onTabChange: (tab: StatementTab) => void;
  onOpenSubmission: (id: number | null) => void;
  previous: ProblemSummary | null;
  next: ProblemSummary | null;
  onNavigate: (problem: ProblemSummary) => void;
}) {
  const hintsRevealed = usePreferences((state) => state.hintsRevealed[problem.id] ?? 0);
  const revealHint = usePreferences((state) => state.revealHint);
  const submissions = useSubmissions(problem.id);
  const submission = useSubmission(submissionId);
  const unlock = useUnlockEditorial(problem.id);
  const reference = useReference(
    problem.id,
    framework,
    problem.editorialUnlocked && tab === "editorial" && !reviewing,
  );

  // Figures in the statement, hints, and editorial all resolve against this problem's assets.
  return (
    <ProblemFigures problemId={problem.id}>
      <Tabs value={tab} onValueChange={(value) => onTabChange(value as StatementTab)} className="h-full">
        <PaneToolbar className="items-stretch justify-between pr-2 pl-6">
          <TabsList>
            <TabsTrigger value="statement">Statement</TabsTrigger>
            <TabsTrigger value="hints">Hints</TabsTrigger>
            <TabsTrigger value="editorial">Editorial</TabsTrigger>
            <TabsTrigger value="submissions">
              Submissions
              {submissions.data && submissions.data.length > 0 ? (
                <span className="tabular text-xs text-faint">{submissions.data.length}</span>
              ) : null}
            </TabsTrigger>
          </TabsList>
          <div className="flex items-center">
            <IconButton
              label={previous ? `Previous: ${previous.title}` : "No previous problem"}
              disabled={!previous}
              onClick={() => previous && onNavigate(previous)}
            >
              <ChevronLeft />
            </IconButton>
            <IconButton
              label={next ? `Next: ${next.title}` : "No next problem"}
              disabled={!next}
              onClick={() => next && onNavigate(next)}
            >
              <ChevronRight />
            </IconButton>
          </div>
        </PaneToolbar>

        <TabsContent value="statement">
          <PageScroll>
            <ProblemHeader problem={problem} track={track} prerequisites={prerequisites} />
            <Prose>{problem.statement}</Prose>
          </PageScroll>
        </TabsContent>

        <TabsContent value="hints">
          <PageScroll>
            {reviewing ? (
              <ReviewHidden what="The hints are" />
            ) : (
              <HintList
                hints={problem.hints}
                revealed={hintsRevealed}
                onReveal={(count) => revealHint(problem.id, count)}
              />
            )}
          </PageScroll>
        </TabsContent>

        <TabsContent value="editorial">
          <PageScroll>
            {reviewing ? (
              <ReviewHidden what="The editorial and the reference solution are" />
            ) : (
              <>
                {unlock.error ? (
                  <ErrorState title="The editorial could not be revealed" error={unlock.error} />
                ) : null}
                <EditorialPanel
                  unlocked={problem.editorialUnlocked}
                  editorial={problem.editorial}
                  reference={reference.data}
                  referenceError={reference.error}
                  framework={framework}
                  revealing={unlock.isPending}
                  onReveal={() => unlock.mutate()}
                />
              </>
            )}
          </PageScroll>
        </TabsContent>

        <TabsContent value="submissions">
          <PageScroll>
            {submissionId !== null ? (
              <SubmissionView
                submission={submission.data?.problemId === problem.id ? submission.data : undefined}
                error={
                  submission.error ??
                  (submission.data && submission.data.problemId !== problem.id
                    ? new Error("That submission belongs to a different problem.")
                    : null)
                }
                onBack={() => onOpenSubmission(null)}
                onRetry={() => void submission.refetch()}
                codeHidden={reviewing}
              />
            ) : submissions.error ? (
              <ErrorState error={submissions.error} onRetry={() => void submissions.refetch()} />
            ) : !submissions.data ? (
              <LoadingLines />
            ) : submissions.data.length === 0 ? (
              <EmptyState title="No submissions yet">
                Submit runs your code against every case, hidden ones included, and keeps a copy of the code
                here.
              </EmptyState>
            ) : (
              <SubmissionList
                className="max-w-measure"
                submissions={submissions.data}
                onOpen={(item) => onOpenSubmission(item.id)}
              />
            )}
          </PageScroll>
        </TabsContent>
      </Tabs>
    </ProblemFigures>
  );
}
