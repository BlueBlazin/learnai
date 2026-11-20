/**
 * Core API types. Mirrors docs/ARCHITECTURE.md section 4 exactly; the Rust side
 * serializes the same shapes with serde (camelCase on the wire).
 */

export type Framework = "torch" | "jax" | "mlx" | "python";
export type Difficulty = "easy" | "medium" | "hard";
export type JudgeKind = "numeric" | "property" | "statistical" | "cost" | "speed" | "simclock" | "service";
export type Requires = "cpu" | "mlx-gpu" | "cuda";
export type ProblemStatus = "todo" | "attempted" | "solved";
export type RunMode = "run" | "submit";
export type CaseStatus = "pass" | "fail" | "error" | "timeout" | "skipped";
export type Verdict =
  | "accepted"
  | "wrong_answer"
  | "runtime_error"
  | "time_limit"
  | "import_error"
  | "harness_error"
  | "canceled"
  /** Nothing that ran failed, but a case was skipped for a model that is not downloaded. */
  | "assets_missing"
  /** Nothing that ran failed, but this machine lacks hardware the problem requires. */
  | "hardware_missing";

export interface Track {
  id: string;
  title: string;
  description: string;
  order: number;
  problemCount: number;
  solvedCount: number;
  /** Slug of the bundled illustration, such as "trailhead". Absent when the module has none. */
  scene?: string;
  /** The module's color as lowercase `#rrggbb`. Absent when the module has none. */
  color?: string;
  /** Ids of the module's core problems, in teaching order. Its other problems are "further". */
  core: string[];
}

/** A subject: an ordered list of modules (`areas/<id>.toml`). */
export interface Area {
  id: string;
  title: string;
  description: string;
  /** Existing track ids only, in order. */
  modules: string[];
}

export interface ProblemSummary {
  id: string;
  title: string;
  track: string;
  order: number;
  difficulty: Difficulty;
  frameworks: Framework[];
  judges: JudgeKind[];
  tags: string[];
  prerequisites: string[];
  requires: Requires;
  status: ProblemStatus;
  solvedFrameworks: Framework[];
  /** Names of the model assets the problem declares. */
  assets: string[];
}

export interface ProblemDetail extends ProblemSummary {
  /** Markdown. */
  statement: string;
  /** Markdown each. */
  hints: string[];
  /** Null while locked. */
  editorial: string | null;
  /** True once solved or explicitly revealed. */
  editorialUnlocked: boolean;
  entry: string[];
  timeLimitS: number;
}

export interface SolutionFile {
  code: string;
  isStarter: boolean;
  path: string;
}

export interface Metric {
  name: string;
  value: number;
  unit?: string;
  limit?: number;
  /** True for metrics where larger is better (throughput): the limit is a minimum. */
  higherIsBetter?: boolean;
}

export interface CaseInfo {
  id: string;
  name: string;
  public: boolean;
  /** Hardware the case needs; absent when it runs everywhere. */
  requires?: Requires;
  /** The stage the case belongs to (capstones); absent for unstaged problems. */
  stage?: string;
}

/** Why a case was not run. */
export type Skip =
  | { kind: "hardware"; requires: Requires }
  | { kind: "asset"; asset: string }
  | { kind: "aborted" };

/** A model asset a run needed and did not find. */
export interface MissingAsset {
  name: string;
  repo: string;
  sizeMb: number;
}

export interface CaseResult {
  id: string;
  status: CaseStatus;
  durationMs: number;
  message: string | null;
  detail: string | null;
  stdout: string;
  metrics: Metric[];
  /** Present when `status` is "skipped" and the judge said why. */
  skip?: Skip;
}

export type RunEvent =
  | { type: "preparing"; message: string }
  | {
      type: "start";
      cases: CaseInfo[];
      env: Record<string, string>;
      /** Hardware the problem requires to be accepted, and whether this machine has it. */
      requires: Requires;
      requiresMet: boolean;
    }
  | { type: "caseStart"; id: string }
  | { type: "caseEnd"; result: CaseResult }
  | { type: "log"; stream: "stdout" | "stderr"; text: string }
  | {
      type: "end";
      verdict: Verdict;
      passed: number;
      total: number;
      durationMs: number;
      metrics: Metric[];
      submissionId: number | null;
      /** Cases that were not run. */
      skipped: number;
      /** With `assets_missing`: the assets to download. */
      missingAssets?: MissingAsset[];
      /** With `hardware_missing`: the hardware this machine lacks. */
      missingHardware?: Requires;
      /**
       * Failure reason and traceback for import and harness failures, or what was
       * skipped and why for the not-judged verdicts (event protocol, section 3.6).
       */
      message: string | null;
      detail: string | null;
    };

/** The review schedule of a solved problem (spaced re-solve, section 4.2). */
export interface Review {
  problemId: string;
  /** 0-based index into the review intervals: the first review is stage 0. */
  stage: number;
  intervalDays: number;
  /** ISO 8601. */
  dueAt: string;
  lastReviewedAt: string | null;
  due: boolean;
}

/** How a review is resolved without solving it. */
export type ReviewAction = "forgot" | "snooze" | "retire";

/** App settings kept by the core. */
export interface Settings {
  /** Whether solved problems are scheduled for review. */
  reviewsEnabled: boolean;
  /** Whether the solution editor offers suggestions and documentation (section 4.4). */
  editorIntelligence: boolean;
}

export type EditorStopReason = "requested" | "replaced" | "disabled" | "shutdown";

/** The state of an editor session's language server. `notInstalled`, `failed`, and `stopped` end the stream. */
export type EditorState =
  | { status: "notInstalled"; message: string }
  | { status: "starting" }
  | {
      status: "ready";
      sessionId: string;
      serverName: string;
      serverVersion: string | null;
      /** `file:///abs/learnai/workspace`, without a trailing slash. Document URIs are formed from it. */
      rootUri: string;
      workspacePath: string;
      /** 1, 2, ... per server process. Every `ready` is a new process that knows nothing. */
      generation: number;
    }
  | { status: "restarting"; attempt: number; maxAttempts: number; delayMs: number; message: string }
  | { status: "failed"; message: string }
  | { status: "stopped"; reason: EditorStopReason };

/** True for the states that end a session's stream: nothing follows them. */
export function isTerminalEditorState(state: EditorState): boolean {
  return state.status === "notInstalled" || state.status === "failed" || state.status === "stopped";
}

export type EditorEvent =
  | { type: "state"; state: EditorState }
  /** One JSON-RPC message from the server, as JSON text. */
  | { type: "message"; message: string };

export interface EditorStatus {
  enabled: boolean;
  installed: boolean;
  installing: boolean;
  serverVersion: string | null;
  /** The latest state of the latest session, or null when none was started. */
  state: EditorState | null;
}

/** What the runtime docs helper knows about a name (section 4.4, `editor_docs`). */
export interface EditorDocs {
  found: boolean;
  /** The public dotted path, for example `torch.nn.functional.softmax`. */
  name: string | null;
  kind: string | null;
  /** `name(parameters) -> result` on one line. */
  signature: string | null;
  /** The docstring as plain text. */
  doc: string | null;
  truncated: boolean;
  /** Why `found` is false. */
  reason: string | null;
}

export interface Submission {
  id: number;
  problemId: string;
  framework: Framework;
  mode: RunMode;
  verdict: Verdict;
  passed: number;
  total: number;
  /** Cases that were not run. */
  skipped: number;
  durationMs: number;
  target: string;
  /** ISO 8601. */
  createdAt: string;
  /** A review attempt, judged from the review file. */
  review: boolean;
}

/** A case result as stored with a submission: the result plus the case's name and visibility. */
export interface StoredCase extends CaseResult {
  name: string;
  public: boolean;
  /** The stage the case belongs to, when the problem has stages. */
  stage?: string;
}

export interface SubmissionDetail extends Submission {
  code: string;
  cases: StoredCase[];
  metrics: Metric[];
}

/** One stage of a study path. `problems` holds existing problem ids only, in study order. */
export interface PathStage {
  title: string;
  description: string;
  problems: string[];
  /** Slug of the bundled illustration, such as "trailhead". */
  scene?: string;
  /** The stage's color as lowercase `#rrggbb`. */
  color?: string;
}

/** A curated, ordered study plan (`paths/<id>.toml`) with progress. */
export interface Path {
  id: string;
  title: string;
  description: string;
  stages: PathStage[];
  problemCount: number;
  solvedCount: number;
}

/** A malformed track, problem, or path that the catalog skipped. */
export interface CatalogIssue {
  path: string;
  message: string;
}

export interface FrameworkEnv {
  framework: Framework;
  installed: boolean;
  version: string | null;
}

export interface EnvStatus {
  uv: string | null;
  python: string | null;
  frameworks: FrameworkEnv[];
  root: string;
}

export interface Stats {
  solved: number;
  total: number;
  byDifficulty: Record<Difficulty, [number, number]>;
  streakDays: number;
  /** Reviews that are due now; 0 while reviews are turned off. */
  reviewsDue: number;
}

export type InstallEvent =
  | { type: "log"; text: string }
  | { type: "done"; ok: boolean; message: string | null };

export type AssetState = "present" | "missing" | "incomplete";

/** A model asset the catalog declares, with its state on disk. */
export interface ModelAsset {
  name: string;
  repo: string;
  revision: string;
  files: string[];
  /** Declared approximate download size. */
  sizeMb: number;
  state: AssetState;
  /** Why it is incomplete. */
  reason: string | null;
  path: string;
  bytesOnDisk: number;
  /** Ids of the problems that declare it. */
  problems: string[];
}

export interface AssetList {
  modelsDir: string;
  assets: ModelAsset[];
}

export type AssetEvent =
  | { type: "preparing"; message: string }
  | { type: "start"; asset: string; totalBytes: number }
  | { type: "progress"; asset: string; file: string; doneBytes: number; totalBytes: number }
  | { type: "assetEnd"; asset: string; ok: boolean; message: string | null }
  | { type: "log"; text: string }
  /** Always last. */
  | { type: "done"; ok: boolean; canceled: boolean; message: string | null };

/** A host alias from `~/.ssh/config`; wildcard patterns are never listed. */
export interface SshHost {
  alias: string;
  hostName: string | null;
  user: string | null;
  port: number | null;
}

export type CheckStepId = "connect" | "uv" | "sync" | "environment" | "doctor";

export interface CheckStep {
  id: CheckStepId;
  ok: boolean;
  message: string;
  /** A command or instruction for the user when the step failed. */
  fix: string | null;
}

/** What `learnai doctor` reported on a target, reduced to what the app shows. */
export interface RemoteDoctor {
  python: string;
  /** `Linux`, `Darwin`. */
  system: string;
  /** `x86_64`, `arm64`. */
  machine: string;
  frameworks: Partial<Record<Framework, string | null>>;
  /** What is present; always includes "cpu". */
  hardware: Requires[];
  gpuName: string | null;
  gpuMemoryBytes: number | null;
}

export interface TargetCheck {
  ok: boolean;
  /** ISO 8601. */
  checkedAt: string;
  /** The steps that ran, in order; the last one is the failing one when `ok` is false. */
  steps: CheckStep[];
  doctor: RemoteDoctor | null;
}

/** A place the judge can run: this computer, or a remote machine reached over SSH. */
export interface TargetInfo {
  /** "local" or "ssh:<alias>". */
  name: string;
  kind: "local" | "ssh";
  label: string;
  host: string | null;
  remoteDir: string | null;
  lastCheck: TargetCheck | null;
}

export type TargetCheckEvent =
  | { type: "step"; step: CheckStep }
  | { type: "log"; text: string }
  /** Always last. */
  | { type: "done"; check: TargetCheck };

export const CHECK_STEPS: readonly CheckStepId[] = ["connect", "uv", "sync", "environment", "doctor"];

export const FRAMEWORKS: readonly Framework[] = ["torch", "jax", "mlx", "python"];
export const DIFFICULTIES: readonly Difficulty[] = ["easy", "medium", "hard"];
export const JUDGES: readonly JudgeKind[] = [
  "numeric",
  "property",
  "statistical",
  "cost",
  "speed",
  "simclock",
  "service",
];
export const STATUSES: readonly ProblemStatus[] = ["todo", "attempted", "solved"];
