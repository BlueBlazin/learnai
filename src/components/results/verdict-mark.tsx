import { cn } from "cn";
import {
  Ban,
  Check,
  CircleCheck,
  CircleX,
  LoaderCircle,
  type LucideIcon,
  Minus,
  OctagonAlert,
  PackageX,
  Timer,
  Unplug,
  X,
} from "lucide-react";
import type { CaseStatus, Verdict } from "@/lib/api";
import { CASE_STATUS_LABEL, CASE_STATUS_TONE, TONE_TEXT, VERDICT_LABEL, VERDICT_TONE } from "@/lib/labels";

const VERDICT_ICON: Record<Verdict, LucideIcon> = {
  accepted: CircleCheck,
  wrong_answer: CircleX,
  runtime_error: OctagonAlert,
  time_limit: Timer,
  import_error: PackageX,
  harness_error: Unplug,
  canceled: Ban,
};

const CASE_ICON: Record<CaseStatus, LucideIcon> = {
  pass: Check,
  fail: X,
  error: OctagonAlert,
  timeout: Timer,
  skipped: Minus,
};

/** A run's verdict as an icon and a word. Each verdict has its own icon, so color is never the only signal. */
export function VerdictMark({
  verdict,
  showLabel = true,
  size = "sm",
  className,
}: {
  verdict: Verdict;
  showLabel?: boolean;
  /** `lg` pairs with display type in a verdict header. */
  size?: "sm" | "lg";
  className?: string;
}) {
  const Icon = VERDICT_ICON[verdict];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-medium whitespace-nowrap",
        TONE_TEXT[VERDICT_TONE[verdict]],
        className,
      )}
    >
      <Icon className={cn("shrink-0", size === "lg" ? "size-4" : "size-3.5")} aria-hidden />
      <span className={showLabel ? undefined : "sr-only"}>{VERDICT_LABEL[verdict]}</span>
    </span>
  );
}

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
