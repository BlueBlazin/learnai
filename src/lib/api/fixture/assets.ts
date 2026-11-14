import type { AssetState, Requires } from "../types";

/** A model the fixture catalog declares, with how it starts out on the fixture disk. */
export interface FixtureAsset {
  name: string;
  repo: string;
  revision: string;
  files: string[];
  /** Declared approximate download size. */
  sizeMb: number;
  /** Exact size of the files, known once a fetch has read the file list. */
  totalBytes: number;
  state: AssetState;
  reason: string | null;
  bytesOnDisk: number;
  /** Simulate one failed attempt (a dropped connection) before a fetch succeeds. */
  failsOnce: boolean;
}

/**
 * One model of each kind: downloaded, not downloaded, and a download that
 * stopped half way. The first two are the real house models.
 */
export function fixtureAssets(): FixtureAsset[] {
  return [
    {
      name: "granite-4.0-1b",
      repo: "ibm-granite/granite-4.0-1b",
      revision: "5d0c3bbf0c1f4c7e9a2f3f0a8f0d6f1f8b1c2d3e",
      files: ["config.json", "tokenizer.json", "model.safetensors"],
      sizeMb: 1630,
      totalBytes: 1_631_244_880,
      state: "incomplete",
      reason: "not downloaded yet: model.safetensors",
      bytesOnDisk: 7_156_949,
      failsOnce: true,
    },
    {
      name: "granite-4.0-350m",
      repo: "ibm-granite/granite-4.0-350m",
      revision: "bd8a1497065c0d6ba1ef19af6b0d2b14bacf71c2",
      files: ["config.json", "tokenizer.json", "model.safetensors"],
      sizeMb: 712,
      totalBytes: 711_966_212,
      state: "present",
      reason: null,
      bytesOnDisk: 711_966_212,
      failsOnce: false,
    },
    {
      name: "granite-4.2-3b-mlx-8bit",
      repo: "lmstudio-community/granite-4.2-3b-MLX-8bit",
      revision: "7c630b526561db6b4641767c54595fe494176695",
      files: ["config.json", "tokenizer.json", "model.safetensors"],
      sizeMb: 3900,
      totalBytes: 3_897_310_556,
      state: "missing",
      reason: null,
      bytesOnDisk: 0,
      failsOnce: false,
    },
  ];
}

/** The hardware of the fixture machine: an Apple silicon Mac. */
export const FIXTURE_HARDWARE: readonly Requires[] = ["cpu", "mlx-gpu"];
