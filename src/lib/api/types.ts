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
  | "canceled";

export interface Track {
  id: string;
  title: string;
  description: string;
  order: number;
  problemCount: number;
  solvedCount: number;
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
}

export interface CaseInfo {
  id: string;
  name: string;
  public: boolean;
}

export interface CaseResult {
  id: string;
  status: CaseStatus;
  durationMs: number;
  message: string | null;
  detail: string | null;
  stdout: string;
  metrics: Metric[];
}

export type RunEvent =
  | { type: "preparing"; message: string }
  | { type: "start"; cases: CaseInfo[]; env: Record<string, string> }
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
      /** Failure reason and traceback for import and harness failures (event protocol, section 3.6). */
      message?: string | null;
      detail?: string | null;
    };

export interface Submission {
  id: number;
  problemId: string;
  framework: Framework;
  mode: RunMode;
  verdict: Verdict;
  passed: number;
  total: number;
  durationMs: number;
  target: string;
  /** ISO 8601. */
  createdAt: string;
}

export interface SubmissionDetail extends Submission {
  code: string;
  cases: CaseResult[];
  metrics: Metric[];
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
}

export type InstallEvent =
  | { type: "log"; text: string }
  | { type: "done"; ok: boolean; message: string | null };

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
