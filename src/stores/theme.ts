import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ThemePreference = "system" | "light" | "dark";
export type ResolvedTheme = "light" | "dark";

interface ThemeStore {
  preference: ThemePreference;
  setPreference: (preference: ThemePreference) => void;
}

/** The key is also read by the inline script in index.html to avoid a flash on launch. */
export const THEME_STORAGE_KEY = "whetstone.theme";

export const useThemeStore = create<ThemeStore>()(
  persist(
    (set) => ({
      preference: "system",
      setPreference: (preference) => set({ preference }),
    }),
    { name: THEME_STORAGE_KEY },
  ),
);

const darkQuery = () => window.matchMedia("(prefers-color-scheme: dark)");

export function resolveTheme(preference: ThemePreference): ResolvedTheme {
  if (preference === "system") return darkQuery().matches ? "dark" : "light";
  return preference;
}

function apply(preference: ThemePreference): void {
  document.documentElement.classList.toggle("dark", resolveTheme(preference) === "dark");
}

/** Keeps the `dark` class on <html> in sync with the preference and the system setting. */
export function installTheme(): () => void {
  apply(useThemeStore.getState().preference);
  const unsubscribe = useThemeStore.subscribe((state) => apply(state.preference));
  const query = darkQuery();
  const onSystemChange = () => apply(useThemeStore.getState().preference);
  query.addEventListener("change", onSystemChange);
  return () => {
    unsubscribe();
    query.removeEventListener("change", onSystemChange);
  };
}

/** The theme in effect, re-rendering when it changes. */
export function useResolvedTheme(): ResolvedTheme {
  const preference = useThemeStore((state) => state.preference);
  return resolveTheme(preference);
}
