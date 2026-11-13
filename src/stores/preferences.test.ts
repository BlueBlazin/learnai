import { describe, expect, it } from "vitest";
import { preferredFramework } from "./preferences";

describe("preferredFramework", () => {
  it("reopens a problem in the framework it was last open in", () => {
    expect(preferredFramework(["torch", "jax", "mlx"], "jax", "mlx")).toBe("jax");
  });

  it("falls back to the framework last worked in when the problem supports it", () => {
    expect(preferredFramework(["torch", "jax", "mlx"], undefined, "mlx")).toBe("mlx");
    expect(preferredFramework(["python"], undefined, "mlx")).toBe("python");
  });

  it("ignores a saved framework the problem no longer supports", () => {
    expect(preferredFramework(["torch", "jax"], "mlx", null)).toBe("torch");
  });

  it("defaults to PyTorch first, whatever order the catalog lists frameworks in", () => {
    expect(preferredFramework(["mlx", "jax", "torch"], undefined, null)).toBe("torch");
    expect(preferredFramework(["mlx", "jax"], undefined, null)).toBe("jax");
    expect(preferredFramework([], undefined, null)).toBeNull();
  });
});
