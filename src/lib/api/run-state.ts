import type {
  CaseInfo,
  CaseResult,
  Framework,
  Metric,
  MissingAsset,
  Requires,
  RunEvent,
  RunMode,
  Verdict,
} from "./types";

/**
 * idle      nothing has run in this workspace yet
 * starting  the start command was sent; no event has arrived
 * preparing the core is syncing the environment or spawning the judge
 * running   cases are executing
 * done      a terminal event arrived, or the run failed to start
 */
export type RunPhase = "idle" | "starting" | "preparing" | "running" | "done";

export interface RunCase {
  info: CaseInfo;
  /** "pending" after the run has ended means the case never ran. */
  state: "pending" | "running" | "done";
  result: CaseResult | null;
}

export interface RunLog {
  stream: "stdout" | "stderr";
  text: string;
}

export interface RunState {
  phase: RunPhase;
  mode: RunMode | null;
  framework: Framework | null;
  /** Where the run executes: "local" or an SSH target's name. */
  target: string | null;
  /** True when the run judges a review attempt. */
  review: boolean;
  runId: string | null;
  preparingMessage: string | null;
  cases: RunCase[];
  env: Record<string, string>;
  logs: RunLog[];
  verdict: Verdict | null;
  passed: number;
  total: number;
  /** Cases that were not run (missing hardware or model, or the run stopped early). */
  skipped: number;
  durationMs: number;
  metrics: Metric[];
  submissionId: number | null;
  /** Hardware the problem requires, and whether this machine has it; null until `start`. */
  requires: Requires | null;
  requiresMet: boolean | null;
  /** With `assets_missing`: the models to download. */
  missingAssets: MissingAsset[];
  /** With `hardware_missing`: what this machine lacks. */
  missingHardware: Requires | null;
  /**
   * Run-level reason and detail sent with `end`: the failure for import and judge
   * failures, what was skipped and why for the not-judged verdicts.
   */
  message: string | null;
  detail: string | null;
  /** Set when the run could not be started at all (transport or core error). */
  startError: string | null;
  canceling: boolean;
}

export type RunAction =
  | { type: "begin"; mode: RunMode; framework: Framework; target?: string; review?: boolean }
  | { type: "started"; runId: string }
  | { type: "event"; event: RunEvent }
  | { type: "failed"; message: string }
  | { type: "canceling" }
  | { type: "reset" };

export const initialRunState: RunState = {
  phase: "idle",
  mode: null,
  framework: null,
  target: null,
  review: false,
  runId: null,
  preparingMessage: null,
  cases: [],
  env: {},
  logs: [],
  verdict: null,
  passed: 0,
  total: 0,
  skipped: 0,
  durationMs: 0,
  metrics: [],
  submissionId: null,
  requires: null,
  requiresMet: null,
  missingAssets: [],
  missingHardware: null,
  message: null,
  detail: null,
  startError: null,
  canceling: false,
};

export function isActive(state: RunState): boolean {
  return state.phase === "starting" || state.phase === "preparing" || state.phase === "running";
}

function upsertCase(cases: RunCase[], id: string, patch: (existing: RunCase) => RunCase): RunCase[] {
  const index = cases.findIndex((c) => c.info.id === id);
  if (index === -1) {
    // The protocol always announces cases in `start`; tolerate a core that does not.
    const fallback: RunCase = {
      info: { id, name: id, public: true },
      state: "pending",
      result: null,
    };
    return [...cases, patch(fallback)];
  }
  return cases.map((c, i) => (i === index ? patch(c) : c));
}

function applyEvent(state: RunState, event: RunEvent): RunState {
  switch (event.type) {
    case "preparing":
      return { ...state, phase: "preparing", preparingMessage: event.message };
    case "start":
      return {
        ...state,
        phase: "running",
        preparingMessage: null,
        env: event.env,
        requires: event.requires,
        requiresMet: event.requiresMet,
        total: event.cases.length,
        cases: event.cases.map((info) => ({ info, state: "pending", result: null })),
      };
    case "caseStart":
      return {
        ...state,
        phase: "running",
        cases: upsertCase(state.cases, event.id, (c) => ({ ...c, state: "running" })),
      };
    case "caseEnd": {
      const cases = upsertCase(state.cases, event.result.id, (c) => ({
        ...c,
        state: "done",
        result: event.result,
      }));
      return {
        ...state,
        phase: "running",
        cases,
        passed: cases.filter((c) => c.result?.status === "pass").length,
        skipped: cases.filter((c) => c.result?.status === "skipped").length,
      };
    }
    case "log":
      return { ...state, logs: [...state.logs, { stream: event.stream, text: event.text }] };
    case "end":
      return {
        ...state,
        phase: "done",
        preparingMessage: null,
        canceling: false,
        verdict: event.verdict,
        passed: event.passed,
        total: event.total,
        skipped: event.skipped,
        durationMs: event.durationMs,
        metrics: event.metrics,
        submissionId: event.submissionId,
        missingAssets: event.missingAssets ?? [],
        missingHardware: event.missingHardware ?? null,
        message: event.message,
        detail: event.detail,
        // A case still marked running when the run ends never produced a result.
        cases: state.cases.map((c) => (c.state === "running" ? { ...c, state: "pending" } : c)),
      };
  }
}

/** Folds the ordered run event stream into what the results panel renders. */
export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "begin":
      return {
        ...initialRunState,
        phase: "starting",
        mode: action.mode,
        framework: action.framework,
        target: action.target ?? "local",
        review: action.review ?? false,
      };
    case "started":
      return state.phase === "idle" ? state : { ...state, runId: action.runId };
    case "event":
      // Nothing follows `end`, and a reset workspace ignores stragglers.
      if (state.phase === "idle" || state.phase === "done") return state;
      return applyEvent(state, action.event);
    case "failed":
      if (state.phase === "idle" || state.phase === "done") return state;
      return {
        ...state,
        phase: "done",
        verdict: "harness_error",
        startError: action.message,
        canceling: false,
      };
    case "canceling":
      return isActive(state) ? { ...state, canceling: true } : state;
    case "reset":
      return initialRunState;
  }
}

/** Text of diagnostic output on one stream, joined for display. */
export function logText(logs: RunLog[], stream?: RunLog["stream"]): string {
  return logs
    .filter((log) => stream === undefined || log.stream === stream)
    .map((log) => log.text)
    .join("\n");
}
