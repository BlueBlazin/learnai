import type { Backend } from "./backend";
import { FixtureBackend } from "./fixture/backend";
import { HttpBackend } from "./http-backend";
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
 */
function selectBackend(): Backend {
  if (isTauri()) return new TauriBackend();
  if (import.meta.env.VITE_BACKEND === "http") {
    return new HttpBackend(import.meta.env.VITE_BACKEND_URL);
  }
  return new FixtureBackend();
}

export const backend: Backend = selectBackend();
