import { Channel, invoke } from "@tauri-apps/api/core";
import type { Backend, StartRunArgs } from "./backend";
import type { AssetEvent, Framework, InstallEvent, RunEvent, TargetCheckEvent } from "./types";

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

  listPaths = () => call<Awaited<ReturnType<Backend["listPaths"]>>>("list_paths");
  getProblemAsset = (id: string, path: string) => call<string>("get_problem_asset", { id, path });

  listAssets = () => call<Awaited<ReturnType<Backend["listAssets"]>>>("list_assets");
  fetchAssets(names: string[], onEvent: (event: AssetEvent) => void, target?: string): Promise<string> {
    const channel = new Channel<AssetEvent>();
    channel.onmessage = onEvent;
    return call<string>("fetch_assets", { names, target: target ?? null, onEvent: channel });
  }
  cancelAssetFetch = (jobId: string) => call<void>("cancel_asset_fetch", { jobId });
  removeAsset = (name: string) => call<Awaited<ReturnType<Backend["removeAsset"]>>>("remove_asset", { name });

  listSshHosts = () => call<Awaited<ReturnType<Backend["listSshHosts"]>>>("list_ssh_hosts");
  listTargets = () => call<Awaited<ReturnType<Backend["listTargets"]>>>("list_targets");
  addTarget = (host: string, remoteDir?: string) =>
    call<Awaited<ReturnType<Backend["addTarget"]>>>("add_target", { host, remoteDir: remoteDir ?? null });
  removeTarget = (name: string) =>
    call<Awaited<ReturnType<Backend["removeTarget"]>>>("remove_target", { name });
  checkTarget(name: string, onEvent: (event: TargetCheckEvent) => void): Promise<string> {
    const channel = new Channel<TargetCheckEvent>();
    channel.onmessage = onEvent;
    return call<string>("check_target", { name, onEvent: channel });
  }
}
