import {
  CHECK_STEPS,
  type CheckStep,
  type CheckStepId,
  type TargetCheck,
  type TargetCheckEvent,
} from "./types";

/** A check of one target, while it runs and after it ends. */
export interface TargetCheckState {
  running: boolean;
  /** The steps that have ended so far, in order. */
  steps: CheckStep[];
  /** Diagnostic output (the environment build), newest last. */
  log: string[];
  /** The result, once `done` has arrived. */
  result: TargetCheck | null;
  /** Set when the check could not be started at all (transport or core error). */
  error: string | null;
}

export type TargetCheckAction =
  | { type: "begin" }
  | { type: "event"; event: TargetCheckEvent }
  | { type: "failed"; message: string };

export const initialTargetCheckState: TargetCheckState = {
  running: false,
  steps: [],
  log: [],
  result: null,
  error: null,
};

/** The most log lines kept: an environment build prints thousands. */
const MAX_LOG_LINES = 400;

/** Folds the ordered check event stream into what a target row renders. */
export function targetCheckReducer(state: TargetCheckState, action: TargetCheckAction): TargetCheckState {
  switch (action.type) {
    case "begin":
      return state.running ? state : { ...initialTargetCheckState, running: true };
    case "failed":
      return state.running ? { ...state, running: false, error: action.message } : state;
    case "event": {
      // Nothing follows `done`.
      if (!state.running) return state;
      const { event } = action;
      switch (event.type) {
        case "step":
          return { ...state, steps: [...state.steps, event.step] };
        case "log":
          return { ...state, log: [...state.log, event.text].slice(-MAX_LOG_LINES) };
        case "done":
          return { ...state, running: false, steps: event.check.steps, result: event.check };
      }
    }
  }
}

export type StepProgress =
  | { id: CheckStepId; state: "done"; step: CheckStep }
  | { id: CheckStepId; state: "running" | "pending" | "skipped" };

/**
 * Every step a check of this kind of target goes through, with where each one
 * stands: ended (with its outcome), running, still to come, or never reached
 * because an earlier step failed.
 */
export function stepProgress(kind: "local" | "ssh", steps: CheckStep[], running: boolean): StepProgress[] {
  const plan = kind === "local" ? CHECK_STEPS.filter((id) => id === "uv" || id === "doctor") : CHECK_STEPS;
  const failed = steps.some((step) => !step.ok);
  let current = running && !failed;
  return plan.map((id) => {
    const step = steps.find((candidate) => candidate.id === id);
    if (step) return { id, state: "done", step };
    if (current) {
      current = false;
      return { id, state: "running" };
    }
    return { id, state: running && !failed ? "pending" : "skipped" };
  });
}
