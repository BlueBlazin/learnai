import type { RunMode, Verdict } from "@/lib/api";
import { formatDuration, formatNumber, pluralize } from "@/lib/format";

export interface RunOutcome {
  verdict: Verdict;
  mode: RunMode | null;
  passed: number;
  total: number;
  /** Cases that were not run. */
  skipped?: number;
  durationMs: number;
}

/** The sentence under a verdict: what happened, in plain words, with the counts. */
export function summarizeRun({ verdict, mode, passed, total, skipped = 0, durationMs }: RunOutcome): string {
  const of = `${passed} of ${pluralize(total, "case")} passed`;
  switch (verdict) {
    case "accepted": {
      const noun = mode === "run" ? "public case" : "case";
      const passedAll = total === 1 ? `The ${noun} passed` : `All ${pluralize(total, noun)} passed`;
      const suffix = mode === "run" ? " Submit to run the hidden cases." : "";
      return `${passedAll} in ${formatDuration(durationMs)}.${suffix}`;
    }
    case "wrong_answer":
      return `${of}.`;
    case "runtime_error":
      return `Your code raised an exception. ${of}.`;
    case "time_limit":
      return `A case ran past its time limit. ${of}.`;
    case "import_error":
      return "Your solution could not be imported, so no cases ran.";
    case "harness_error":
      return "The judge did not start. This is a problem with the environment, not with your code.";
    case "canceled":
      return total === 0
        ? "Canceled before any case ran."
        : `${passed} of ${pluralize(total, "case")} had passed when you canceled.`;
    case "assets_missing":
    case "hardware_missing":
      return `${of}, ${formatNumber(skipped)} skipped.`;
  }
}

/**
 * Why a not-judged outcome is not a solve, in one sentence. Stored submissions
 * keep no run-level message, so history says this where a live run shows the
 * judge's own explanation.
 */
export function notJudgedReason(verdict: "assets_missing" | "hardware_missing"): string {
  return verdict === "assets_missing"
    ? "Nothing that ran failed. The skipped cases need a model that was not downloaded, so the solution was not judged on them."
    : "Nothing that ran failed. The skipped cases need hardware the machine it ran on does not have; the solution can only be accepted on a machine that runs every case.";
}

/** "Python 3.15.2, torch 2.15.0, mps" from the run's environment record. */
export function describeEnv(env: Record<string, string>): string {
  return Object.entries(env)
    .map(([key, value]) =>
      key === "device" || key === "platform" ? value : `${key === "python" ? "Python" : key} ${value}`,
    )
    .join(", ");
}
