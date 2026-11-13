import type {
  CatalogIssue,
  EnvStatus,
  Framework,
  InstallEvent,
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

  /** Malformed tracks and problems the catalog skipped. Empty when everything loaded. */
  listCatalogIssues(): Promise<CatalogIssue[]>;
}

export const LOCAL_TARGET = "local";
