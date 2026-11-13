import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ThemePreference = "system" | "light" | "dark";
type ResolvedTheme = "light" | "dark";

interface ThemeStore {
  preference: ThemePreference;
  setPreference: (preference: ThemePreference) => void;
}

/** The key is also read by the inline script in index.html to avoid a flash on launch. */
const THEME_STORAGE_KEY = "whetstone.theme";

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

function resolveTheme(preference: ThemePreference): ResolvedTheme {
  if (preference === "system") return darkQuery().matches ? "dark" : "light";
  return preference;
}

/**
 * Switches the theme in one frame. Controls transition their colors on hover
 * (120ms), and a transition on an inherited color restarts at every level of
 * nesting, so without this a theme change would ripple through the page for
 * over a second. Transitions are off only while the new colors are applied.
 */
function apply(preference: ThemePreference): void {
  const root = document.documentElement;
  const dark = resolveTheme(preference) === "dark";
  if (root.classList.contains("dark") === dark) return;
  root.classList.add("theme-switching");
  root.classList.toggle("dark", dark);
  // Reading a computed style forces the new colors to resolve while transitions are off.
  void window.getComputedStyle(root).color;
  requestAnimationFrame(() => requestAnimationFrame(() => root.classList.remove("theme-switching")));
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

/** The theme in effect, re-rendering when the preference changes. */
export function useResolvedTheme(): ResolvedTheme {
  return resolveTheme(useThemeStore((state) => state.preference));
}
