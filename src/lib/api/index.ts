import type { Backend } from "./backend";
import { HttpBackend } from "./http-backend";
import { lazyBackend, unavailableBackend } from "./lazy-backend";
import { TauriBackend } from "./tauri-backend";

export type { Backend, StartRunArgs } from "./backend";
export { LOCAL_TARGET } from "./backend";
export * from "./types";

export function isTauri(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/**
 * Backend selection (docs/ARCHITECTURE.md section 2): Tauri when running in the
 * shell, the devbridge when `VITE_BACKEND=http`, fixtures otherwise.
 *
 * The fixture data is large and only useful for component work, so it loads on
 * demand, and builds made for the Tauri shell (`__LEARNAI_FIXTURES__` false,
 * see vite.config.ts) leave it out entirely.
 */
function selectBackend(): Backend {
  if (isTauri()) return new TauriBackend();
  if (import.meta.env.VITE_BACKEND === "http") {
    return new HttpBackend(import.meta.env.VITE_BACKEND_URL);
  }
  if (__LEARNAI_FIXTURES__) {
    return lazyBackend("fixture", async () => new (await import("./fixture/backend")).FixtureBackend());
  }
  return unavailableBackend(
    "fixture",
    "This build runs only inside the Learn AI app. Open the app, or start the development server.",
  );
}

export const backend: Backend = selectBackend();
