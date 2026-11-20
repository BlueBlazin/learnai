import type { RemoteDoctor, Requires, TargetInfo } from "@/lib/api";
import { FRAMEWORK_LABEL } from "@/lib/labels";

export const LOCAL = "local";

/** The host alias in an SSH target's name ("ssh:gpu-box"), or null for anything else. */
export function hostOf(name: string): string | null {
  return name.startsWith("ssh:") && name.length > 4 ? name.slice(4) : null;
}

/** A target's name as a person says it: the host alias, or "this computer". */
export function targetLabel(name: string): string {
  return hostOf(name) ?? (name === LOCAL ? "this computer" : name);
}

const SYSTEM_LABEL: Record<string, string> = { Darwin: "macOS", Linux: "Linux" };

/** "Linux, x86_64" or "macOS, arm64" from a doctor report. */
export function describeMachine(doctor: RemoteDoctor): string {
  return [SYSTEM_LABEL[doctor.system] ?? doctor.system, doctor.machine].filter(Boolean).join(", ");
}

/** Memory the way hardware is sold: 25,757,220,864 bytes is "24 GB". */
export function formatMemory(bytes: number): string {
  const gib = bytes / 2 ** 30;
  return gib >= 1 ? `${Math.round(gib)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`;
}

/** "NVIDIA GeForce RTX 4090, 24 GB", or null when the machine has no GPU the judge can use. */
export function describeGpu(doctor: RemoteDoctor): string | null {
  if (!doctor.gpuName) return doctor.hardware.length > 1 ? "Present" : null;
  return doctor.gpuMemoryBytes ? `${doctor.gpuName}, ${formatMemory(doctor.gpuMemoryBytes)}` : doctor.gpuName;
}

/**
 * One line about a target: where it runs and, once a check has looked, its
 * chip and memory. "Runs here. Apple M3 Max, 64 GB"; "Over SSH" before any check.
 */
export function summarizeTarget(target: TargetInfo): string {
  const where = target.kind === "local" ? "Runs here" : "Over SSH";
  const doctor = target.lastCheck?.doctor;
  if (!doctor) return where;
  const gpu = describeGpu(doctor);
  return `${where}. ${gpu && gpu !== "Present" ? gpu : describeMachine(doctor)}`;
}

/** The installed tensor frameworks with their versions, in the canonical order. */
export function installedFrameworks(doctor: RemoteDoctor): { label: string; version: string }[] {
  return (["torch", "jax", "mlx"] as const).flatMap((framework) => {
    const version = doctor.frameworks[framework];
    return version ? [{ label: FRAMEWORK_LABEL[framework], version }] : [];
  });
}

/** The hardware a target is known to have from its last successful look, or null when unknown. */
export function knownHardware(target: TargetInfo | undefined): Requires[] | null {
  return target?.lastCheck?.doctor?.hardware ?? null;
}

/** The remote targets whose last check found `requires`. */
export function targetsWith(targets: TargetInfo[], requires: Requires): TargetInfo[] {
  return targets.filter((target) => target.kind === "ssh" && knownHardware(target)?.includes(requires));
}

/** The target to run on: the remembered one if it still exists, otherwise this computer. */
export function resolveTarget(targets: TargetInfo[] | undefined, preferred: string): string {
  return targets?.some((target) => target.name === preferred) ? preferred : LOCAL;
}
