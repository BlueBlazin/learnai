import { create } from "zustand";
import { persist } from "zustand/middleware";
import { FRAMEWORKS, type Framework } from "@/lib/api";
import { isCollapsedSize, type WorkspacePanel } from "@/lib/workspace-layout";

interface PreferencesStore {
  /** The framework each problem was last open in. */
  frameworkByProblem: Record<string, Framework>;
  /**
   * The framework the user last worked in (edited or ran code), on any problem;
   * the default for a problem not opened before. Looking at another framework's
   * tab does not change it.
   */
  lastFramework: Framework | null;
  /** How many hints have been revealed per problem. */
  hintsRevealed: Record<string, number>;
  /** The run target last chosen in a workspace: "local" or an SSH target's name. */
  target: string;
  setTarget: (target: string) => void;
  /**
   * The size (percent of its group) each collapsible workspace panel last showed
   * at: what it reopens to. Whether it is collapsed is in the saved panel layout.
   */
  panelSizes: Partial<Record<WorkspacePanel, number>>;
  /** Remembers a panel's size while it is showing; a collapsed size is ignored. */
  setPanelSize: (panel: WorkspacePanel, size: number | undefined) => void;
  /** Remembers the framework a problem is open in. */
  setFramework: (problemId: string, framework: Framework) => void;
  /** Records real work in a framework: an edit, a run, or a submit. */
  markFrameworkUsed: (problemId: string, framework: Framework) => void;
  revealHint: (problemId: string, count: number) => void;
  /** How many problems the learner aims to solve each week. */
  weeklyGoal: number;
  /** Sets the weekly goal, kept to a whole number from 1 through 21. */
  setWeeklyGoal: (goal: number) => void;
}

export const MIN_WEEKLY_GOAL = 1;
export const MAX_WEEKLY_GOAL = 21;
export const DEFAULT_WEEKLY_GOAL = 5;

/** A goal as a whole number within the allowed range; a value that is not a number gives the default. */
export function clampWeeklyGoal(goal: number): number {
  if (!Number.isFinite(goal)) return DEFAULT_WEEKLY_GOAL;
  return Math.min(MAX_WEEKLY_GOAL, Math.max(MIN_WEEKLY_GOAL, Math.round(goal)));
}

/** Per-user UI state that should survive a restart. Not progress: that lives in the core. */
export const usePreferences = create<PreferencesStore>()(
  persist(
    (set, get) => ({
      frameworkByProblem: {},
      lastFramework: null,
      hintsRevealed: {},
      target: "local",
      setTarget: (target) => set({ target }),
      panelSizes: {},
      setPanelSize: (panel, size) => {
        if (size === undefined || isCollapsedSize(size) || get().panelSizes[panel] === size) return;
        set((state) => ({ panelSizes: { ...state.panelSizes, [panel]: size } }));
      },
      setFramework: (problemId, framework) =>
        set((state) => ({ frameworkByProblem: { ...state.frameworkByProblem, [problemId]: framework } })),
      markFrameworkUsed: (problemId, framework) => {
        const state = get();
        if (state.lastFramework === framework && state.frameworkByProblem[problemId] === framework) return;
        set({
          frameworkByProblem: { ...state.frameworkByProblem, [problemId]: framework },
          lastFramework: framework,
        });
      },
      revealHint: (problemId, count) =>
        set((state) => ({ hintsRevealed: { ...state.hintsRevealed, [problemId]: count } })),
      weeklyGoal: DEFAULT_WEEKLY_GOAL,
      setWeeklyGoal: (goal) => set({ weeklyGoal: clampWeeklyGoal(goal) }),
    }),
    {
      name: "learnai.preferences",
      // `panelSizes` and `weeklyGoal` were added without a new version: a stored state that lacks them keeps
      // the default above, and a version bump would run `migrate` and forget `lastFramework`.
      version: 1,
      // Version 0 set `lastFramework` on every switch, so one look at another framework
      // became the default everywhere. Drop that value; it is re-learned from real work.
      migrate: (persisted) => ({ ...(persisted as Partial<PreferencesStore>), lastFramework: null }),
    },
  ),
);

/**
 * Chooses the framework to open a problem in: the one this problem was last
 * open in; otherwise the one the user last worked in, if the problem supports
 * it; otherwise the first it supports in the canonical order (PyTorch first),
 * whatever order the catalog lists them in.
 */
export function preferredFramework(
  supported: Framework[],
  saved: Framework | undefined,
  last: Framework | null,
): Framework | null {
  if (saved && supported.includes(saved)) return saved;
  if (last && supported.includes(last)) return last;
  return FRAMEWORKS.find((framework) => supported.includes(framework)) ?? supported[0] ?? null;
}
