import type { EnvStatus, Framework } from "@/lib/api";

/** What stands between the user and a successful run, most fundamental first. */
export type EnvironmentProblem =
  | { kind: "uv-missing" }
  | { kind: "not-set-up" }
  | { kind: "framework-missing"; framework: Framework };

/** Where uv's own installation instructions live. */
export const UV_INSTALL_URL = "https://docs.astral.sh/uv/getting-started/installation/";

/**
 * Decides what, if anything, is wrong with the Python environment for a run in
 * `framework`. Unknown status (still loading, or the probe failed) is not a
 * problem: the run itself reports what goes wrong.
 */
export function environmentProblem(
  env: EnvStatus | undefined,
  framework: Framework | null,
): EnvironmentProblem | null {
  if (!env) return null;
  if (env.uv === null) return { kind: "uv-missing" };
  if (env.python === null) return { kind: "not-set-up" };
  const entryFor = (name: Framework) => env.frameworks.find((candidate) => candidate.framework === name);
  // PyTorch is a base dependency of the judge. If it is missing, the environment was
  // never synced (uv can report a Python version from an empty virtual environment).
  if (entryFor("torch")?.installed === false) return { kind: "not-set-up" };
  if (framework === null) return null;
  return entryFor(framework)?.installed === false ? { kind: "framework-missing", framework } : null;
}
