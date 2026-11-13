import { describe, expect, it, vi } from "vitest";
import type { Backend } from "./backend";
import { lazyBackend, unavailableBackend } from "./lazy-backend";

describe("lazyBackend", () => {
  it("loads the implementation once, on first use, and forwards calls with their arguments", async () => {
    const getProblem = vi.fn(async (id: string) => ({ id }));
    const load = vi.fn(async () => ({ kind: "fixture", getProblem }) as unknown as Backend);
    const backend = lazyBackend("fixture", load);

    expect(backend.kind).toBe("fixture");
    expect(load).not.toHaveBeenCalled();

    await expect(backend.getProblem("a")).resolves.toEqual({ id: "a" });
    await expect(backend.getProblem("b")).resolves.toEqual({ id: "b" });
    expect(load).toHaveBeenCalledTimes(1);
    expect(getProblem.mock.calls).toEqual([["a"], ["b"]]);
  });

  it("is not mistaken for a promise", async () => {
    const backend = lazyBackend("fixture", async () => ({}) as Backend);
    expect(await Promise.resolve(backend)).toBe(backend);
  });

  it("rejects every call when no backend is available", async () => {
    const backend = unavailableBackend("fixture", "No data source in this build.");
    await expect(backend.listProblems()).rejects.toThrow("No data source in this build.");
    await expect(backend.envStatus()).rejects.toThrow("No data source in this build.");
  });
});
