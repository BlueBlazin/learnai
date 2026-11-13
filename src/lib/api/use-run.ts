import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useReducer, useRef } from "react";
import { LOCAL_TARGET } from "./backend";
import { backend } from "./index";
import { invalidateProgress } from "./queries";
import { initialRunState, isActive, type RunState, runReducer } from "./run-state";
import type { Framework, RunMode } from "./types";

export interface RunController {
  state: RunState;
  active: boolean;
  start: (framework: Framework, mode: RunMode) => Promise<void>;
  cancel: () => void;
}

/**
 * Owns one problem's run: starts it, folds the event stream into `RunState`,
 * cancels on request or when the workspace goes away, and refreshes progress
 * queries after a stored submission.
 */
export function useRun(problemId: string): RunController {
  const client = useQueryClient();
  const [state, dispatch] = useReducer(runReducer, initialRunState);
  /** Identifies the current run so events from an abandoned one are dropped. */
  const generation = useRef(0);
  const live = useRef<{ runId: string | null; cancelRequested: boolean } | null>(null);

  const abandon = useCallback(() => {
    const run = live.current;
    live.current = null;
    generation.current += 1;
    if (run?.runId) void backend.cancelRun(run.runId).catch(() => {});
    else if (run) run.cancelRequested = true;
  }, []);

  // A different problem, or unmount: stop whatever is running and start clean.
  // biome-ignore lint/correctness/useExhaustiveDependencies: problemId is the trigger.
  useEffect(() => {
    dispatch({ type: "reset" });
    return abandon;
  }, [problemId, abandon]);

  const start = useCallback(
    async (framework: Framework, mode: RunMode) => {
      if (live.current) return;
      const token = ++generation.current;
      const run = { runId: null as string | null, cancelRequested: false };
      live.current = run;
      dispatch({ type: "begin", mode, framework });
      try {
        const runId = await backend.startRun(
          { problemId, framework, mode, target: LOCAL_TARGET },
          (event) => {
            if (generation.current !== token) return;
            dispatch({ type: "event", event });
            if (event.type === "end") {
              live.current = null;
              if (mode === "submit") void invalidateProgress(client, problemId);
            }
          },
        );
        run.runId = runId;
        if (run.cancelRequested) {
          void backend.cancelRun(runId).catch(() => {});
        }
        if (generation.current === token) dispatch({ type: "started", runId });
      } catch (error) {
        if (generation.current !== token) return;
        live.current = null;
        dispatch({ type: "failed", message: error instanceof Error ? error.message : String(error) });
      }
    },
    [client, problemId],
  );

  const cancel = useCallback(() => {
    const run = live.current;
    if (!run) return;
    dispatch({ type: "canceling" });
    if (run.runId) void backend.cancelRun(run.runId).catch(() => {});
    else run.cancelRequested = true;
  }, []);

  return { state, active: isActive(state), start, cancel };
}
