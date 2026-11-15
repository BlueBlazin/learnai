import { describe, expect, it } from "vitest";
import { missingHardware } from "./hardware";

const NO_RUN = { requires: null, requiresMet: null };

describe("missingHardware", () => {
  it("never reports a CPU problem", () => {
    expect(missingHardware("cpu", { run: NO_RUN, isMac: true })).toBeNull();
    expect(missingHardware("cpu", { run: { requires: "cpu", requiresMet: true }, isMac: false })).toBeNull();
  });

  it("knows a Mac has no NVIDIA GPU before anything has run", () => {
    expect(missingHardware("cuda", { run: NO_RUN, isMac: true })).toBe("cuda");
    expect(missingHardware("cuda", { run: NO_RUN, isMac: false })).toBeNull();
    expect(missingHardware("mlx-gpu", { run: NO_RUN, isMac: true })).toBeNull();
    // Unknown is not reported as missing.
    expect(missingHardware("mlx-gpu", { run: NO_RUN, isMac: false })).toBeNull();
  });

  it("lets the judge's own answer decide once a run has started", () => {
    expect(
      missingHardware("mlx-gpu", { run: { requires: "mlx-gpu", requiresMet: false }, isMac: true }),
    ).toBe("mlx-gpu");
    // A run that found the hardware (or was told to pretend it did) overrides the guess.
    expect(missingHardware("cuda", { run: { requires: "cuda", requiresMet: true }, isMac: true })).toBeNull();
    expect(
      missingHardware("cuda", {
        run: { requires: "cuda", requiresMet: true },
        lastVerdict: "hardware_missing",
        isMac: false,
      }),
    ).toBeNull();
  });

  it("falls back to the most recent submission", () => {
    expect(missingHardware("cuda", { run: NO_RUN, lastVerdict: "hardware_missing", isMac: false })).toBe(
      "cuda",
    );
    expect(missingHardware("cuda", { run: NO_RUN, lastVerdict: "wrong_answer", isMac: false })).toBeNull();
  });

  it("uses what a check of the machine found when nothing has run", () => {
    expect(missingHardware("cuda", { run: NO_RUN, isMac: false, known: ["cpu", "cuda"] })).toBeNull();
    expect(missingHardware("cuda", { run: NO_RUN, isMac: false, known: ["cpu"] })).toBe("cuda");
    expect(missingHardware("mlx-gpu", { run: NO_RUN, isMac: true, known: null })).toBeNull();
  });
});
