import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Framework } from "@/lib/api";

interface PreferencesStore {
  /** The framework last used on each problem. */
  frameworkByProblem: Record<string, Framework>;
  /** The framework last chosen anywhere; the default for a problem not opened before. */
  lastFramework: Framework | null;
  /** How many hints have been revealed per problem. */
  hintsRevealed: Record<string, number>;
  setFramework: (problemId: string, framework: Framework) => void;
  revealHint: (problemId: string, count: number) => void;
}

/** Per-user UI state that should survive a restart. Not progress: that lives in the core. */
export const usePreferences = create<PreferencesStore>()(
  persist(
    (set) => ({
      frameworkByProblem: {},
      lastFramework: null,
      hintsRevealed: {},
      setFramework: (problemId, framework) =>
        set((state) => ({
          frameworkByProblem: { ...state.frameworkByProblem, [problemId]: framework },
          lastFramework: framework,
        })),
      revealHint: (problemId, count) =>
        set((state) => ({ hintsRevealed: { ...state.hintsRevealed, [problemId]: count } })),
    }),
    { name: "whetstone.preferences" },
  ),
);

/** Chooses the framework to open a problem in. */
export function preferredFramework(
  supported: Framework[],
  saved: Framework | undefined,
  last: Framework | null,
): Framework | null {
  if (saved && supported.includes(saved)) return saved;
  if (last && supported.includes(last)) return last;
  return supported[0] ?? null;
}
