import { cn } from "cn";
import { Circle, CircleCheck, CircleX, LoaderCircle, type LucideIcon } from "lucide-react";

export type CheckStepState = "pass" | "fail" | "running" | "pending" | "skipped";

const ICON: Record<CheckStepState, LucideIcon> = {
  pass: CircleCheck,
  fail: CircleX,
  running: LoaderCircle,
  pending: Circle,
  skipped: Circle,
};

const TONE: Record<CheckStepState, string> = {
  pass: "text-pass",
  fail: "text-fail",
  running: "animate-spin text-primary",
  pending: "text-faint",
  skipped: "text-faint",
};

const WORD: Record<CheckStepState, string> = {
  pass: "Passed",
  fail: "Failed",
  running: "Running",
  pending: "Waiting",
  skipped: "Not reached",
};

/**
 * Where one step of a target check stands, as a shape: a check, a cross, a
 * spinner, or an empty ring for a step that has not run. The word is read by
 * screen readers; the row beside the mark carries it for everyone else.
 */
export function CheckStepMark({ state, className }: { state: CheckStepState; className?: string }) {
  const Icon = ICON[state];
  return (
    <span className={cn("inline-flex shrink-0", className)}>
      <Icon className={cn("size-3.5", TONE[state])} aria-hidden />
      <span className="sr-only">{WORD[state]}</span>
    </span>
  );
}
