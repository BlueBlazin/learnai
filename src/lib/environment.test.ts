import { describe, expect, it } from "vitest";
import type { EnvStatus } from "@/lib/api";
import { environmentProblem } from "./environment";

const ready: EnvStatus = {
  uv: "0.13.8",
  python: "3.15.2",
  root: "/r",
  frameworks: [
    { framework: "torch", installed: true, version: "2.15.0" },
    { framework: "jax", installed: false, version: null },
    { framework: "python", installed: true, version: "3.15.2" },
  ],
};

describe("environmentProblem", () => {
  it("reports nothing while the status is unknown or everything is in place", () => {
    expect(environmentProblem(undefined, "torch")).toBeNull();
    expect(environmentProblem(ready, "torch")).toBeNull();
    expect(environmentProblem(ready, "python")).toBeNull();
    expect(environmentProblem(ready, null)).toBeNull();
    // A framework the status does not mention is not known to be missing.
    expect(environmentProblem(ready, "mlx")).toBeNull();
  });

  it("puts a missing uv before everything else", () => {
    const env = { ...ready, uv: null, python: null };
    expect(environmentProblem(env, "jax")).toEqual({ kind: "uv-missing" });
  });

  it("reports an environment that was never synced", () => {
    const env = { ...ready, python: null };
    expect(environmentProblem(env, "torch")).toEqual({ kind: "not-set-up" });
  });

  it("treats a missing base dependency as an environment that was never synced", () => {
    const env: EnvStatus = {
      ...ready,
      frameworks: [
        { framework: "torch", installed: false, version: null },
        { framework: "jax", installed: false, version: null },
        { framework: "python", installed: true, version: "3.15.2" },
      ],
    };
    expect(environmentProblem(env, "jax")).toEqual({ kind: "not-set-up" });
    expect(environmentProblem(env, "python")).toEqual({ kind: "not-set-up" });
  });

  it("names the framework that is not installed", () => {
    expect(environmentProblem(ready, "jax")).toEqual({ kind: "framework-missing", framework: "jax" });
  });
});
