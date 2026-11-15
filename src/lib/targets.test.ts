import { describe, expect, it } from "vitest";
import type { RemoteDoctor, TargetInfo } from "@/lib/api";
import {
  describeGpu,
  describeMachine,
  formatMemory,
  hostOf,
  installedFrameworks,
  knownHardware,
  resolveTarget,
  targetLabel,
  targetsWith,
} from "./targets";

const doctor = (over: Partial<RemoteDoctor> = {}): RemoteDoctor => ({
  python: "3.15.2",
  system: "Linux",
  machine: "x86_64",
  frameworks: { torch: "2.15.0", jax: null, mlx: null },
  hardware: ["cpu", "cuda"],
  gpuName: "NVIDIA GeForce RTX 4090",
  gpuMemoryBytes: 25_757_220_864,
  ...over,
});

const target = (name: string, report: RemoteDoctor | null): TargetInfo => ({
  name,
  kind: name === "local" ? "local" : "ssh",
  label: name,
  host: hostOf(name),
  remoteDir: name === "local" ? null : "~/.whetstone",
  lastCheck: report ? { ok: true, checkedAt: "2026-11-15T10:00:00.000Z", steps: [], doctor: report } : null,
});

describe("target names", () => {
  it("reads the host alias out of an SSH target name", () => {
    expect(hostOf("ssh:gpu-box")).toBe("gpu-box");
    expect(hostOf("local")).toBeNull();
    expect(hostOf("ssh:")).toBeNull();
    expect(targetLabel("ssh:gpu-box")).toBe("gpu-box");
    expect(targetLabel("local")).toBe("this computer");
  });

  it("falls back to this computer when the remembered target is gone", () => {
    const targets = [target("local", null), target("ssh:gpu-box", null)];
    expect(resolveTarget(targets, "ssh:gpu-box")).toBe("ssh:gpu-box");
    expect(resolveTarget(targets, "ssh:removed")).toBe("local");
    expect(resolveTarget(undefined, "ssh:gpu-box")).toBe("local");
  });
});

describe("doctor reports", () => {
  it("describes the machine and its GPU", () => {
    expect(describeMachine(doctor())).toBe("Linux, x86_64");
    expect(describeMachine(doctor({ system: "Darwin", machine: "arm64" }))).toBe("macOS, arm64");
    expect(describeGpu(doctor())).toBe("NVIDIA GeForce RTX 4090, 24 GB");
    expect(describeGpu(doctor({ gpuMemoryBytes: null }))).toBe("NVIDIA GeForce RTX 4090");
    expect(describeGpu(doctor({ gpuName: null, hardware: ["cpu"] }))).toBeNull();
    expect(describeGpu(doctor({ gpuName: null }))).toBe("Present");
    expect(formatMemory(68_719_476_736)).toBe("64 GB");
    expect(formatMemory(536_870_912)).toBe("512 MB");
  });

  it("lists only the frameworks that are installed", () => {
    expect(
      installedFrameworks(doctor({ frameworks: { torch: "2.15.0", jax: "0.12.0.1", mlx: null } })),
    ).toEqual([
      { label: "PyTorch", version: "2.15.0" },
      { label: "JAX", version: "0.12.0.1" },
    ]);
    expect(installedFrameworks(doctor({ frameworks: {} }))).toEqual([]);
  });

  it("finds the remote machines known to have some hardware", () => {
    const targets = [
      target("local", doctor({ hardware: ["cpu", "cuda"] })),
      target("ssh:gpu-box", doctor()),
      target("ssh:cpu-box", doctor({ hardware: ["cpu"], gpuName: null })),
      target("ssh:unchecked", null),
    ];
    expect(targetsWith(targets, "cuda").map((found) => found.name)).toEqual(["ssh:gpu-box"]);
    expect(targetsWith(targets, "mlx-gpu")).toEqual([]);
    expect(knownHardware(targets[3])).toBeNull();
    expect(knownHardware(undefined)).toBeNull();
  });
});
