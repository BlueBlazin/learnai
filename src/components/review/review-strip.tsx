import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { Notice } from "@/components/app/notice";
import { Button } from "@/components/ui/button";
import type { Review } from "@/lib/api";
import { reviewLabel } from "@/lib/reviews";

/**
 * The strip under the editor toolbar while a review is in progress. It says
 * what this is, and offers every way out: give up and look ("I forgot"), put it
 * off a day, start the attempt over, stop reviewing the problem for good, or
 * just leave and come back later. Solving it is the other way out: an accepted
 * Submit ends the review by itself.
 */
export function ReviewStrip({
  review,
  busy = false,
  error,
  onForgot,
  onSnooze,
  onStartOver,
  onRetire,
  onLeave,
}: {
  review: Review;
  /** True while a run or one of these actions is in flight: nothing here can be pressed. */
  busy?: boolean;
  /** Why the last action failed, in the core's words. */
  error?: string | null;
  onForgot: () => void;
  onSnooze: () => void;
  onStartOver: () => void;
  onRetire: () => void;
  onLeave: () => void;
}) {
  return (
    <Notice
      tone={error ? "fail" : "info"}
      layout="strip"
      action={
        <Button variant="outline" size="xs" disabled={busy} onClick={onLeave}>
          Leave review
        </Button>
      }
      footer={
        <div className="-ml-2 flex flex-wrap items-center gap-x-1 pl-5.5 text-muted-foreground">
          <ConfirmDialog
            trigger={
              <Button variant="ghost" size="xs" disabled={busy}>
                I forgot
              </Button>
            }
            title="Show your solution?"
            description="This ends the review and shows your saved solution, the hints, and the editorial again. The schedule starts over, and the problem comes back for review tomorrow."
            confirmLabel="Show my solution"
            onConfirm={onForgot}
          />
          <Button variant="ghost" size="xs" disabled={busy} onClick={onSnooze}>
            Snooze a day
          </Button>
          <ConfirmDialog
            trigger={
              <Button variant="ghost" size="xs" disabled={busy}>
                Start over
              </Button>
            }
            title="Start the review over?"
            description="Your review attempt will be replaced with the blank starter. Your saved solution is not touched."
            confirmLabel="Start over"
            destructive
            onConfirm={onStartOver}
          />
          <ConfirmDialog
            trigger={
              <Button variant="ghost" size="xs" disabled={busy}>
                Stop reviewing this problem
              </Button>
            }
            title="Stop reviewing this problem?"
            description="It will not come up for review again. Your solution and your submissions stay as they are."
            confirmLabel="Stop reviewing"
            destructive
            onConfirm={onRetire}
          />
        </div>
      }
    >
      {error ? (
        error
      ) : (
        <>
          <span className="font-medium">{reviewLabel(review)}.</span> Solve it again from the blank starter.
        </>
      )}
    </Notice>
  );
}
