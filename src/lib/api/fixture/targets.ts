import type { CheckStep, RemoteDoctor, SshHost, TargetCheck, TargetInfo } from "../types";

/** The fixture `~/.ssh/config`: two hosts that are already targets and one that is not. */
export const FIXTURE_SSH_HOSTS: SshHost[] = [
  { alias: "gpu-box", hostName: "10.0.0.5", user: "ada", port: null },
  { alias: "studio", hostName: "studio.local", user: null, port: null },
  { alias: "lab-a100", hostName: "a100.lab.example.org", user: "ada", port: 2222 },
];

const LOCAL_DOCTOR: RemoteDoctor = {
  python: "3.15.2",
  system: "Darwin",
  machine: "arm64",
  frameworks: { torch: "2.15.0", jax: "0.12.0.1", mlx: null },
  hardware: ["cpu", "mlx-gpu"],
  gpuName: "Apple M3 Max",
  gpuMemoryBytes: 68_719_476_736,
};

const GPU_BOX_DOCTOR: RemoteDoctor = {
  python: "3.15.2",
  system: "Linux",
  machine: "x86_64",
  frameworks: { torch: "2.15.0", jax: "0.12.0.1", mlx: null },
  hardware: ["cpu", "cuda"],
  gpuName: "NVIDIA GeForce RTX 4090",
  gpuMemoryBytes: 25_757_220_864,
};

const A100_DOCTOR: RemoteDoctor = {
  ...GPU_BOX_DOCTOR,
  frameworks: { torch: "2.15.0", jax: null, mlx: null },
  gpuName: "NVIDIA A100-SXM4-80GB",
  gpuMemoryBytes: 85_899_345_920,
};

const pass = (id: CheckStep["id"], message: string): CheckStep => ({ id, ok: true, message, fix: null });

/** What a check of each fixture machine finds: its steps, and the report when it gets that far. */
export function fixtureCheckScript(target: TargetInfo): { steps: CheckStep[]; doctor: RemoteDoctor | null } {
  if (target.kind === "local") {
    return {
      steps: [pass("uv", "Found uv 0.13.8."), pass("doctor", "Python 3.15.2, Apple M3 Max.")],
      doctor: LOCAL_DOCTOR,
    };
  }
  const host = target.host ?? target.label;
  if (host === "studio") {
    return {
      steps: [
        pass("connect", "Connected to studio (Darwin arm64)."),
        {
          id: "uv",
          ok: false,
          message:
            "uv is not installed on `studio`. Install it with uv's official installer, then try again:",
          fix: "ssh studio 'curl -LsSf https://astral.sh/uv/install.sh | sh'",
        },
      ],
      doctor: null,
    };
  }
  const doctor = host === "gpu-box" ? GPU_BOX_DOCTOR : A100_DOCTOR;
  const root = `/home/ada/${(target.remoteDir ?? "~/.learnai").replace(/^~\//, "")}/root`;
  return {
    steps: [
      pass("connect", `Connected to ${host} (Linux x86_64).`),
      pass("uv", "Found uv at /home/ada/.local/bin/uv."),
      pass("sync", `Copied the judge to ${root}/python.`),
      pass("environment", "The Python environment is ready."),
      pass("doctor", `Python ${doctor.python}, ${doctor.gpuName}.`),
    ],
    doctor,
  };
}

function checkOf(target: TargetInfo, checkedAt: string): TargetCheck {
  const { steps, doctor } = fixtureCheckScript(target);
  return { ok: steps.every((step) => step.ok), checkedAt, steps, doctor };
}

/**
 * This computer, a healthy remote machine with an NVIDIA GPU, and one whose
 * check stops at uv with the command that fixes it.
 */
export function fixtureTargets(now: Date): TargetInfo[] {
  const ago = (minutes: number) => new Date(now.getTime() - minutes * 60_000).toISOString();
  const bare = (name: string, kind: TargetInfo["kind"], label: string): TargetInfo => ({
    name,
    kind,
    label,
    host: kind === "ssh" ? label : null,
    remoteDir: kind === "ssh" ? "~/.learnai" : null,
    lastCheck: null,
  });
  const targets = [
    bare("local", "local", "This Mac"),
    bare("ssh:gpu-box", "ssh", "gpu-box"),
    bare("ssh:studio", "ssh", "studio"),
  ];
  const checkedAt = [ago(12), ago(95), ago(60 * 26)];
  return targets.map((target, index) => ({
    ...target,
    lastCheck: checkOf(target, checkedAt[index] ?? ago(0)),
  }));
}
