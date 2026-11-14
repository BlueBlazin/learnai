import type {
  AssetEvent,
  AssetList,
  CatalogIssue,
  EnvStatus,
  Framework,
  InstallEvent,
  Path,
  ProblemDetail,
  ProblemSummary,
  RunEvent,
  RunMode,
  SolutionFile,
  Stats,
  Submission,
  SubmissionDetail,
  Track,
} from "./types";

export interface StartRunArgs {
  problemId: string;
  framework: Framework;
  mode: RunMode;
  /** Execution target. Only "local" exists until the SSH target lands. */
  target: string;
}

/**
 * The single seam between the UI and whetstone-core. One method per command in
 * docs/ARCHITECTURE.md section 4 (camelCase here, snake_case on the wire).
 * Every method rejects with an `Error` whose message is the core's error string.
 */
export interface Backend {
  readonly kind: "tauri" | "http" | "fixture";

  listTracks(): Promise<Track[]>;
  listProblems(): Promise<ProblemSummary[]>;
  getProblem(id: string): Promise<ProblemDetail>;

  readSolution(id: string, framework: Framework): Promise<SolutionFile>;
  writeSolution(id: string, framework: Framework, code: string): Promise<void>;
  resetSolution(id: string, framework: Framework): Promise<SolutionFile>;

  /**
   * Starts a run and resolves with its id as soon as the core accepts it.
   * Events are delivered in order until a single `end` event; nothing follows it.
   */
  startRun(args: StartRunArgs, onEvent: (event: RunEvent) => void): Promise<string>;
  cancelRun(runId: string): Promise<void>;

  listSubmissions(problemId?: string): Promise<Submission[]>;
  getSubmission(id: number): Promise<SubmissionDetail>;

  unlockEditorial(id: string): Promise<ProblemDetail>;
  getReference(id: string, framework: Framework): Promise<string>;

  envStatus(): Promise<EnvStatus>;
  /** Resolves once the install job is accepted; progress arrives as events, ending with `done`. */
  envInstall(framework: Framework, onEvent: (event: InstallEvent) => void): Promise<void>;

  getStats(): Promise<Stats>;

  /** Malformed tracks, problems, and paths the catalog skipped. Empty when everything loaded. */
  listCatalogIssues(): Promise<CatalogIssue[]>;

  /** Study paths, in file-name order, with progress. */
  listPaths(): Promise<Path[]>;

  /**
   * A statement figure as a `data:` URL. `path` is relative to the problem's
   * `assets/` directory and may not escape it; only `.svg` and `.png` are served.
   */
  getProblemAsset(id: string, path: string): Promise<string>;

  /** Every model asset the catalog declares, with its state on disk. */
  listAssets(): Promise<AssetList>;
  /**
   * Starts downloading the named assets and resolves with the job id as soon as
   * the core accepts it. Events are delivered in order until a single `done`
   * event; nothing follows it. One fetch runs at a time: a second call rejects.
   */
  fetchAssets(names: string[], onEvent: (event: AssetEvent) => void): Promise<string>;
  /** Stops a fetch; files that were complete are kept. Unknown or finished jobs are ignored. */
  cancelAssetFetch(jobId: string): Promise<void>;
  /** Deletes a declared asset from disk and resolves with the new listing. */
  removeAsset(name: string): Promise<AssetList>;
}

export const LOCAL_TARGET = "local";
