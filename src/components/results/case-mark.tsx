import { cn } from "cn";
import { Check, LoaderCircle, type LucideIcon, Minus, OctagonAlert, Timer, X } from "lucide-react";
import type { CaseStatus } from "@/lib/api";
import { CASE_STATUS_LABEL, CASE_STATUS_TONE, TONE_TEXT } from "@/lib/labels";

const CASE_ICON: Record<CaseStatus, LucideIcon> = {
  pass: Check,
  fail: X,
  error: OctagonAlert,
  timeout: Timer,
  skipped: Minus,
};

/** One case's state as an icon: a status once finished, a spinner while running, a dot while waiting. */
export function CaseMark({
  status,
  className,
}: {
  status: CaseStatus | "running" | "pending" | "not-run";
  className?: string;
}) {
  if (status === "running") {
    return (
      <LoaderCircle
        className={cn("size-3.5 shrink-0 animate-spin text-primary", className)}
        aria-label="Running"
      />
    );
  }
  if (status === "pending" || status === "not-run") {
    return (
      <span
        className={cn("flex size-3.5 shrink-0 items-center justify-center", className)}
        role="img"
        aria-label={status === "pending" ? "Waiting" : "Not run"}
      >
        <span className="size-1 rounded-full bg-faint" />
      </span>
    );
  }
  const Icon = CASE_ICON[status];
  return (
    <Icon
      className={cn("size-3.5 shrink-0", TONE_TEXT[CASE_STATUS_TONE[status]], className)}
      aria-label={CASE_STATUS_LABEL[status]}
    />
  );
}
