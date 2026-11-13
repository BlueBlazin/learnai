import { create } from "zustand";

interface CommandPaletteStore {
  open: boolean;
  setOpen: (open: boolean) => void;
}

/** Whether the command palette is open; the app bar and Mod-K both drive it. */
export const useCommandPalette = create<CommandPaletteStore>()((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
}));
