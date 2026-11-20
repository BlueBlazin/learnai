import type {
  Area,
  AssetEvent,
  AssetList,
  CatalogIssue,
  EditorDocs,
  EditorEvent,
  EditorStatus,
  EnvStatus,
  Framework,
  InstallEvent,
  Path,
  ProblemDetail,
  ProblemSummary,
  Review,
  ReviewAction,
  RunEvent,
  RunMode,
  Settings,
  SolutionFile,
  SshHost,
  Stats,
  Submission,
  SubmissionDetail,
  TargetCheckEvent,
  TargetInfo,
  Track,
} from "./types";

export interface StartRunArgs {
  problemId: string;
  framework: Framework;
  mode: RunMode;
  /** Where the judge runs: "local" or the name of an SSH target ("ssh:<alias>"). */
  target: string;
  /** Judge the review attempt instead of the saved solution. */
  review?: boolean;
}

/**
 * The single seam between the UI and learnai-core. One method per command in
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

  /** Malformed tracks, problems, areas, and paths the catalog skipped. Empty when everything loaded. */
  listCatalogIssues(): Promise<CatalogIssue[]>;

  /** Areas, in file-name order. Each lists existing track ids only. */
  listAreas(): Promise<Area[]>;

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
   * `target` is where the files should land: this computer by default, or an SSH
   * target, where the same fetch runs on the remote machine.
   */
  fetchAssets(names: string[], onEvent: (event: AssetEvent) => void, target?: string): Promise<string>;
  /** Stops a fetch; files that were complete are kept. Unknown or finished jobs are ignored. */
  cancelAssetFetch(jobId: string): Promise<void>;
  /** Deletes a declared asset from disk and resolves with the new listing. */
  removeAsset(name: string): Promise<AssetList>;

  /** Host aliases from the user's SSH configuration (no wildcard patterns). */
  listSshHosts(): Promise<SshHost[]>;
  /** Run targets: `local` first, then the SSH targets in the order they were added. */
  listTargets(): Promise<TargetInfo[]>;
  /** Registers an SSH host from the configuration as a target; `remoteDir` defaults to `~/.learnai`. */
  addTarget(host: string, remoteDir?: string): Promise<TargetInfo[]>;
  /** Forgets an SSH target. Nothing is deleted on the remote machine. */
  removeTarget(name: string): Promise<TargetInfo[]>;
  /**
   * Starts checking a target and resolves with the job id as soon as the core
   * accepts it. Each step arrives as it ends; a single `done` event is last and
   * carries the result, which the core also saves as the target's `lastCheck`.
   */
  checkTarget(name: string, onEvent: (event: TargetCheckEvent) => void): Promise<string>;

  /** Active review schedules, soonest due first. Empty while reviews are turned off. */
  listReviews(): Promise<Review[]>;
  /** Discards the review attempt and resolves with the starter. The saved solution is not touched. */
  startReview(id: string, framework: Framework): Promise<SolutionFile>;
  /** The review attempt, or the starter when there is none. */
  readReviewSolution(id: string, framework: Framework): Promise<SolutionFile>;
  writeReviewSolution(id: string, framework: Framework, code: string): Promise<void>;
  /** Ends a review without solving it and resolves with the remaining schedules. */
  resolveReview(id: string, action: ReviewAction): Promise<Review[]>;

  getSettings(): Promise<Settings>;
  /** Turns reviews on or off. Schedules are kept while they are off. */
  setReviewsEnabled(enabled: boolean): Promise<Settings>;

  // Editor intelligence (docs/ARCHITECTURE.md section 4.4).

  /** Turns suggestions and documentation on or off. Off stops the server before resolving. */
  setEditorIntelligence(enabled: boolean): Promise<Settings>;
  /** Whether the language server is enabled and installed, and its latest state. Starts nothing. */
  editorStatus(): Promise<EditorStatus>;
  /**
   * Starts an editor session, replacing a running one, and resolves with its id.
   * Events arrive in order and end with exactly one terminal state (`notInstalled`,
   * `failed`, or `stopped`). `ready` may arrive before this resolves.
   */
  editorStart(onEvent: (event: EditorEvent) => void): Promise<string>;
  /** Sends one JSON-RPC message (as JSON text) to the session's server. Rejects when it is not running. */
  editorSend(sessionId: string, message: string): Promise<void>;
  /** Stops that session, or whatever runs when no id is given. Unknown sessions are ignored. */
  editorStop(sessionId?: string): Promise<void>;
  /** Installs the language server. Progress arrives as events, ending with `done`, like `envInstall`. */
  editorInstall(onEvent: (event: InstallEvent) => void): Promise<void>;
  /**
   * Looks a name up in the installed libraries. `expr` is a dotted name as the
   * file spells it; with `member`, the lookup is for that attribute of `expr`.
   */
  editorDocs(source: string, expr: string, member?: string): Promise<EditorDocs>;
}

export const LOCAL_TARGET = "local";
