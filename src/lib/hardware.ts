import type { Requires, Verdict } from "@/lib/api";

/** What is known about this machine's hardware from the workspace's runs. */
export interface HardwareEvidence {
  /** The last run's `start` event: what the problem requires, and whether it was met. Null before any run. */
  run: { requires: Requires | null; requiresMet: boolean | null };
  /** The verdict of the most recent stored submission, if there is one. */
  lastVerdict?: Verdict;
  /** True on Apple hardware. */
  isMac: boolean;
}

/**
 * The hardware a problem requires that this machine does not have, or null.
 *
 * The judge is the authority: once a run has started, its `requiresMet` decides.
 * Before that, the most recent submission decides if it ended with missing
 * hardware, and a problem that requires CUDA cannot be met on a Mac. Anything
 * else is unknown, and unknown is not reported as missing.
 */
export function missingHardware(
  requires: Requires,
  { run, lastVerdict, isMac }: HardwareEvidence,
): Exclude<Requires, "cpu"> | null {
  if (requires === "cpu") return null;
  if (run.requiresMet !== null) return run.requiresMet ? null : requires;
  if (lastVerdict === "hardware_missing") return requires;
  return requires === "cuda" && isMac ? requires : null;
}
