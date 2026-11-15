import { create } from "zustand";
import { backend } from "./index";
import { keys, queryClient } from "./queries";
import {
  initialTargetCheckState,
  type TargetCheckAction,
  type TargetCheckState,
  targetCheckReducer,
} from "./target-check-state";
import type { TargetInfo } from "./types";

interface TargetCheckStore {
  checks: Record<string, TargetCheckState>;
  dispatch: (name: string, action: TargetCheckAction) => void;
}

/**
 * One store for the whole app: a check started in Settings keeps running, and
 * stays visible, while the user is in a workspace.
 */
const useTargetCheckStore = create<TargetCheckStore>()((set) => ({
  checks: {},
  dispatch: (name, action) =>
    set((state) => ({
      checks: {
        ...state.checks,
        [name]: targetCheckReducer(state.checks[name] ?? initialTargetCheckState, action),
      },
    })),
}));

function check(name: string): void {
  const { checks, dispatch } = useTargetCheckStore.getState();
  if (checks[name]?.running) return;
  dispatch(name, { type: "begin" });
  backend
    .checkTarget(name, (event) => {
      dispatch(name, { type: "event", event });
      if (event.type === "done") {
        // The core saved the result as the target's last check: show it everywhere.
        queryClient.setQueryData<TargetInfo[]>(keys.targets, (targets) =>
          targets?.map((target) => (target.name === name ? { ...target, lastCheck: event.check } : target)),
        );
        void queryClient.invalidateQueries({ queryKey: keys.targets });
      }
    })
    .catch((error: unknown) => {
      dispatch(name, { type: "failed", message: error instanceof Error ? error.message : String(error) });
    });
}

export interface TargetCheckController {
  /** Per-target check state, by target name. Absent: no check has run in this session. */
  checks: Record<string, TargetCheckState>;
  /** Starts checking a target; a check already running for it is left alone. */
  check: (name: string) => void;
}

/** Target checks: start one and follow its steps. Shared by every screen. */
export function useTargetCheck(): TargetCheckController {
  const checks = useTargetCheckStore((state) => state.checks);
  return { checks, check };
}
