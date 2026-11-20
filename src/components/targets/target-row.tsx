import { useState } from "react";
import { ConfirmDialog } from "@/components/app/confirm-dialog";
import { Notice } from "@/components/app/notice";
import { OutputBlock } from "@/components/results/output-block";
import { RowStatus } from "@/components/settings/row-status";
import { SettingRow } from "@/components/settings/setting-row";
import { Button } from "@/components/ui/button";
import type { TargetInfo } from "@/lib/api";
import { stepProgress, type TargetCheckState } from "@/lib/api/target-check-state";
import { formatRelativeTime } from "@/lib/format";
import { summarizeTarget } from "@/lib/targets";
import { CheckStepList } from "./check-step-list";
import { TargetSummary } from "./target-summary";

/**
 * One run target in Settings, on one line: its name, where it runs and what it
 * is, when it was last checked, and the Check and Remove actions. Behind the
 * name: the machine as the last check found it and that check's steps. The
 * details open by themselves while a check runs and whenever the last one
 * failed, so the failing step, its reason, and the fix are never a click away.
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
  const failed = last?.ok === false;
  const steps = running
    ? stepProgress(target.kind, check?.steps ?? [], true)
    : last
      ? stepProgress(target.kind, last.steps, false)
      : null;

  // Open while a check runs and when the last one failed. Details that opened by
  // themselves close again once the check passes; ones the user opened stay.
  const needsOpen = running || failed;
  const [open, setOpen] = useState(needsOpen);
  const [openedItself, setOpenedItself] = useState(needsOpen);
  const [needed, setNeeded] = useState(needsOpen);
  if (needed !== needsOpen) {
    setNeeded(needsOpen);
    if (needsOpen && !open) {
      setOpen(true);
      setOpenedItself(true);
    } else if (!needsOpen && openedItself) {
      setOpen(false);
      setOpenedItself(false);
    }
  }
  const toggle = (next: boolean) => {
    setOpen(next);
    setOpenedItself(false);
  };

  const summary = <TargetSummary doctor={last?.doctor ?? null} remoteDir={target.remoteDir} />;
  const hasDetails = steps !== null || last?.doctor != null || Boolean(target.remoteDir);
  return (
    <SettingRow
      label={target.label}
      help={summarizeTarget(target)}
      status={
        running ? (
          <RowStatus tone="busy">Checking</RowStatus>
        ) : last ? (
          <RowStatus tone={last.ok ? "pass" : "fail"}>
            {last.ok ? "Checked" : "Check failed"} {formatRelativeTime(last.checkedAt)}
          </RowStatus>
        ) : (
          <RowStatus tone="idle">Not checked</RowStatus>
        )
      }
      body={check?.error ? <Notice tone="fail">{check.error}</Notice> : null}
      open={open}
      onOpenChange={toggle}
      details={
        hasDetails ? (
          <>
            {summary}
            {steps ? <CheckStepList steps={steps} /> : null}
            {running && check && check.log.length > 0 ? (
              <OutputBlock label="Check log">{check.log.join("\n")}</OutputBlock>
            ) : null}
          </>
        ) : undefined
      }
    >
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
          description={`Learn AI will stop offering ${target.label} as a place to run. Nothing is deleted on the remote machine: the files in ${target.remoteDir} stay until you delete them there, and you can add it again at any time.`}
          confirmLabel="Remove target"
          onConfirm={onRemove}
        />
      ) : null}
      <Button variant="outline" size="xs" onClick={onCheck} disabled={running || removing}>
        {running ? "Checking" : last ? "Check again" : "Check"}
      </Button>
    </SettingRow>
  );
}
