import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { Notice } from "@/components/app/notice";
import { OutputBlock } from "@/components/results/output-block";
import { Button } from "@/components/ui/button";
import type { TargetInfo } from "@/lib/api";
import { stepProgress, type TargetCheckState } from "@/lib/api/target-check-state";
import { formatRelativeTime } from "@/lib/format";
import { CheckStepList } from "./check-step-list";
import { TargetSummary } from "./target-summary";

/**
 * One run target in Settings: what it is, what its last check found, and the
 * Check and Remove actions. While a check runs its steps appear one by one; a
 * check that stopped keeps the failing step, its reason, and the fix on screen.
 */
export function TargetRow({
  target,
  check,
  removing = false,
  onCheck,
  onRemove,
}: {
  target: TargetInfo;
  /** The check running now, or the last one of this session. */
  check?: TargetCheckState;
  /** True while this target is being removed. */
  removing?: boolean;
  onCheck: () => void;
  /** Absent for this computer, which cannot be removed. */
  onRemove?: () => void;
}) {
  const running = check?.running ?? false;
  const last = target.lastCheck;
  // While a check runs, show its steps as they end; otherwise only a check that failed needs them.
  const steps = running
    ? stepProgress(target.kind, check?.steps ?? [], true)
    : last && !last.ok
      ? stepProgress(target.kind, last.steps, false)
      : null;
  const doctor = last?.doctor ?? null;
  return (
    <li className="flex flex-col gap-2.5 py-4">
      <div className="flex items-start justify-between gap-6">
        <div className="flex min-w-0 flex-col gap-1.5">
          <p className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <span className="selectable text-sm font-medium">{target.label}</span>
            <span className="text-xs text-muted-foreground">
              {target.kind === "local" ? (
                "Runs here"
              ) : (
                <>
                  Over SSH, in <code className="selectable font-mono">{target.remoteDir}</code>
                </>
              )}
            </span>
          </p>
          {doctor ? (
            <TargetSummary doctor={doctor} />
          ) : running || last ? null : (
            <p className="text-xs text-muted-foreground">
              {target.kind === "local"
                ? "Not checked yet. Check to see what this computer has."
                : "Not checked yet. Check to set it up and see what it has; a run does this too."}
            </p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button variant="outline" size="xs" onClick={onCheck} disabled={running || removing}>
            {running ? "Checking" : last ? "Check again" : "Check"}
          </Button>
          {onRemove ? (
            <ConfirmDialog
              trigger={
                <Button
                  variant="ghost"
                  size="xs"
                  className="text-muted-foreground"
                  disabled={running || removing}
                >
                  Remove
                </Button>
              }
              title={`Remove ${target.label}?`}
              description={`Whetstone will stop offering ${target.label} as a place to run. Nothing is deleted on the remote machine: the files in ${target.remoteDir} stay until you delete them there, and you can add it again at any time.`}
              confirmLabel="Remove target"
              onConfirm={onRemove}
            />
          ) : null}
        </div>
      </div>

      {steps ? <CheckStepList steps={steps} /> : null}
      {running && check && check.log.length > 0 ? (
        <OutputBlock label="Check log">{check.log.join("\n")}</OutputBlock>
      ) : null}
      {check?.error ? <Notice tone="fail">{check.error}</Notice> : null}
      {last && !running ? (
        <p className="text-xs text-muted-foreground">
          {last.ok ? "Checked" : "Check failed"} {formatRelativeTime(last.checkedAt)}
        </p>
      ) : null}
    </li>
  );
}
