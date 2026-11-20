import type {
  AssetState,
  CaseStatus,
  Difficulty,
  Framework,
  JudgeKind,
  ProblemStatus,
  Requires,
  Verdict,
} from "@/lib/api";
import type { CoreFilter } from "@/lib/filters";
import type { SettingsSection } from "@/lib/routes";

/**
 * How an outcome is colored. Color never carries the meaning alone: every use pairs
 * it with an icon and a word. `plain` is body ink: an outcome that is neither a
 * success nor a failure (a run that could not judge every case). `neutral` is
 * secondary ink: nothing happened (canceled, skipped).
 */
export type Tone = "pass" | "fail" | "warn" | "plain" | "neutral";

export const FRAMEWORK_LABEL: Record<Framework, string> = {
  torch: "PyTorch",
  jax: "JAX",
  mlx: "MLX",
  python: "Python",
};

export const DIFFICULTY_LABEL: Record<Difficulty, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

export const DIFFICULTY_LEVEL: Record<Difficulty, 1 | 2 | 3> = { easy: 1, medium: 2, hard: 3 };

export const STATUS_LABEL: Record<ProblemStatus, string> = {
  todo: "Not started",
  attempted: "Attempted",
  solved: "Solved",
};

export const CORE_FILTER_LABEL: Record<CoreFilter, string> = {
  core: "Core only",
  further: "Further only",
};

export const JUDGE_LABEL: Record<JudgeKind, string> = {
  numeric: "Numeric",
  property: "Property",
  statistical: "Statistical",
  cost: "Cost",
  speed: "Speed",
  simclock: "Simulated clock",
  service: "Service",
};

export const JUDGE_DESCRIPTION: Record<JudgeKind, string> = {
  numeric: "Output must match the reference within a float32 tolerance.",
  property: "An invariant must hold on seeded random inputs.",
  statistical: "The output distribution must match the target on fixed seeds.",
  cost: "An instrumented object counts the work done; the count has a limit.",
  speed: "Timed against the reference in the same process; never an absolute threshold.",
  simclock: "Runs in a discrete-event simulation with virtual time.",
  service: "An HTTP client exercises your app on a local port.",
};

export const REQUIRES_LABEL: Record<Requires, string> = {
  cpu: "CPU",
  "mlx-gpu": "Apple GPU",
  cuda: "CUDA GPU",
};

/** Hardware as the object of "needs": "This problem needs an NVIDIA GPU (CUDA)". */
export const REQUIRES_PHRASE: Record<Requires, string> = {
  cpu: "a CPU",
  "mlx-gpu": "an Apple GPU that MLX can use",
  cuda: "an NVIDIA GPU (CUDA)",
};

export const ASSET_STATE_LABEL: Record<AssetState, string> = {
  present: "Downloaded",
  missing: "Not downloaded",
  incomplete: "Incomplete",
};

export const SETTINGS_SECTION_LABEL: Record<SettingsSection, string> = {
  general: "General",
  environment: "Environment",
  models: "Models",
  targets: "Run targets",
  files: "Files",
};

export const VERDICT_LABEL: Record<Verdict, string> = {
  accepted: "Accepted",
  wrong_answer: "Wrong answer",
  runtime_error: "Runtime error",
  time_limit: "Time limit exceeded",
  import_error: "Import error",
  harness_error: "The judge could not run",
  canceled: "Canceled",
  assets_missing: "Not judged: model not downloaded",
  hardware_missing: "Not judged: hardware missing",
};

/** Verdicts that end a run without judging every case: neither a success nor a failure. */
export function isNotJudged(verdict: Verdict): verdict is "assets_missing" | "hardware_missing" {
  return verdict === "assets_missing" || verdict === "hardware_missing";
}

export const VERDICT_TONE: Record<Verdict, Tone> = {
  accepted: "pass",
  wrong_answer: "fail",
  runtime_error: "fail",
  time_limit: "warn",
  import_error: "fail",
  harness_error: "warn",
  canceled: "neutral",
  assets_missing: "plain",
  hardware_missing: "plain",
};

export const CASE_STATUS_LABEL: Record<CaseStatus, string> = {
  pass: "Passed",
  fail: "Failed",
  error: "Error",
  timeout: "Timed out",
  skipped: "Skipped",
};

export const CASE_STATUS_TONE: Record<CaseStatus, Tone> = {
  pass: "pass",
  fail: "fail",
  error: "fail",
  timeout: "warn",
  skipped: "neutral",
};

/** Tailwind text color class per tone. */
export const TONE_TEXT: Record<Tone, string> = {
  pass: "text-pass",
  fail: "text-fail",
  warn: "text-warn",
  plain: "text-foreground",
  neutral: "text-muted-foreground",
};
