import { describe, expect, it } from "vitest";
import { clampWeeklyGoal, preferredFramework, usePreferences } from "./preferences";

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

describe("weekly goal", () => {
  it("defaults to 5", () => {
    expect(usePreferences.getState().weeklyGoal).toBe(5);
  });

  it("is set, rounded, and kept within 1 through 21", () => {
    const { setWeeklyGoal } = usePreferences.getState();
    setWeeklyGoal(10);
    expect(usePreferences.getState().weeklyGoal).toBe(10);
    setWeeklyGoal(0);
    expect(usePreferences.getState().weeklyGoal).toBe(1);
    setWeeklyGoal(99);
    expect(usePreferences.getState().weeklyGoal).toBe(21);
    setWeeklyGoal(6.6);
    expect(usePreferences.getState().weeklyGoal).toBe(7);
    setWeeklyGoal(5);
  });

  it("falls back to the default for a value that is not a number", () => {
    expect(clampWeeklyGoal(Number.NaN)).toBe(5);
  });
});
