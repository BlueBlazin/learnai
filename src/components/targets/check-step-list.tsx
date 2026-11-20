import { cn } from "cn";
import type { CheckStepId } from "@/lib/api";
import type { StepProgress } from "@/lib/api/target-check-state";
import { CheckStepMark, type CheckStepState } from "./check-step-mark";
import { FixText } from "./fix-text";

const STEP_LABEL: Record<CheckStepId, string> = {
  connect: "Connect",
  uv: "Find uv",
  sync: "Copy the judge",
  environment: "Build the environment",
  doctor: "Read the hardware",
};

function stateOf(step: StepProgress): CheckStepState {
  if (step.state === "done") return step.step.ok ? "pass" : "fail";
  return step.state;
}

/**
 * The steps of a target check, in order, each with its mark and what it found.
 * A step that failed says why and, when there is something the user can do,
 * gives it as text to copy. Steps after a failure are shown as not reached.
 */
export function CheckStepList({ steps, className }: { steps: StepProgress[]; className?: string }) {
  return (
    <ol className={cn("flex flex-col gap-1.5 text-xs", className)} aria-label="Check steps">
      {steps.map((entry) => {
        const state = stateOf(entry);
        const done = entry.state === "done" ? entry.step : null;
        return (
          <li key={entry.id} className="flex flex-col gap-1.5">
            <p className="flex items-baseline gap-2">
              <CheckStepMark state={state} className="translate-y-0.5" />
              <span
                className={cn(
                  // With the mark, as wide as the label column of `MetaList` rows: the two line up in a target's details.
                  "w-32.5 shrink-0",
                  state === "pending" || state === "skipped" ? "text-muted-foreground" : "text-foreground",
                )}
              >
                {STEP_LABEL[entry.id]}
              </span>
              {done ? (
                <span className={cn("selectable min-w-0", done.ok ? "text-muted-foreground" : "text-fail")}>
                  {done.message}
                </span>
              ) : null}
            </p>
            {done?.fix ? (
              <div className="pl-5.5">
                <FixText>{done.fix}</FixText>
              </div>
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}
