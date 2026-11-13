import { Channel, invoke } from "@tauri-apps/api/core";
import type { Backend, StartRunArgs } from "./backend";
import type { Framework, InstallEvent, RunEvent } from "./types";

/** Tauri rejects with the command's error string; normalize to Error. */
async function call<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  try {
    return await invoke<T>(command, args);
  } catch (cause) {
    throw new Error(typeof cause === "string" ? cause : String((cause as Error)?.message ?? cause));
  }
}

/** Production transport: one Tauri command per row of the core API table. */
export class TauriBackend implements Backend {
  readonly kind = "tauri" as const;

  listTracks = () => call<Awaited<ReturnType<Backend["listTracks"]>>>("list_tracks");
  listProblems = () => call<Awaited<ReturnType<Backend["listProblems"]>>>("list_problems");
  getProblem = (id: string) => call<Awaited<ReturnType<Backend["getProblem"]>>>("get_problem", { id });

  readSolution = (id: string, framework: Framework) =>
    call<Awaited<ReturnType<Backend["readSolution"]>>>("read_solution", { id, framework });
  writeSolution = (id: string, framework: Framework, code: string) =>
    call<void>("write_solution", { id, framework, code });
  resetSolution = (id: string, framework: Framework) =>
    call<Awaited<ReturnType<Backend["resetSolution"]>>>("reset_solution", { id, framework });

  startRun(args: StartRunArgs, onEvent: (event: RunEvent) => void): Promise<string> {
    const channel = new Channel<RunEvent>();
    channel.onmessage = onEvent;
    return call<string>("start_run", { ...args, onEvent: channel });
  }
  cancelRun = (runId: string) => call<void>("cancel_run", { runId });

  listSubmissions = (problemId?: string) =>
    call<Awaited<ReturnType<Backend["listSubmissions"]>>>("list_submissions", {
      problemId: problemId ?? null,
    });
  getSubmission = (id: number) =>
    call<Awaited<ReturnType<Backend["getSubmission"]>>>("get_submission", { id });

  unlockEditorial = (id: string) =>
    call<Awaited<ReturnType<Backend["unlockEditorial"]>>>("unlock_editorial", { id });
  getReference = (id: string, framework: Framework) => call<string>("get_reference", { id, framework });

  envStatus = () => call<Awaited<ReturnType<Backend["envStatus"]>>>("env_status");
  async envInstall(framework: Framework, onEvent: (event: InstallEvent) => void): Promise<void> {
    const channel = new Channel<InstallEvent>();
    channel.onmessage = onEvent;
    await call<unknown>("env_install", { framework, onEvent: channel });
  }

  getStats = () => call<Awaited<ReturnType<Backend["getStats"]>>>("get_stats");

  listCatalogIssues = () => call<Awaited<ReturnType<Backend["listCatalogIssues"]>>>("list_catalog_issues");
}
