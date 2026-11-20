import { create } from "zustand";

export interface WorkspaceActions {
  problemTitle: string;
  run: () => void;
  submit: () => void;
  busy: boolean;
  /** Whether each collapsible panel is hidden, so the palette can word its toggle. */
  statementCollapsed: boolean;
  resultsCollapsed: boolean;
  toggleStatement: () => void;
  toggleResults: () => void;
}

interface WorkspaceActionsStore {
  actions: WorkspaceActions | null;
  setActions: (actions: WorkspaceActions | null) => void;
}

/**
 * The open workspace registers its Run, Submit, and panel toggles here so the
 * command palette can offer them.
 */
export const useWorkspaceActions = create<WorkspaceActionsStore>()((set) => ({
  actions: null,
  setActions: (actions) => set({ actions }),
}));
