import { create } from "zustand";

export interface WorkspaceActions {
  problemTitle: string;
  run: () => void;
  submit: () => void;
  busy: boolean;
}

interface WorkspaceActionsStore {
  actions: WorkspaceActions | null;
  setActions: (actions: WorkspaceActions | null) => void;
}

/** The open workspace registers its Run and Submit here so the command palette can offer them. */
export const useWorkspaceActions = create<WorkspaceActionsStore>()((set) => ({
  actions: null,
  setActions: (actions) => set({ actions }),
}));

interface PaletteStore {
  open: boolean;
  setOpen: (open: boolean) => void;
}

export const useCommandPalette = create<PaletteStore>()((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
}));
