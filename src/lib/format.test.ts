import { describe, expect, it } from "vitest";
import {
  formatBytes,
  formatDuration,
  formatMetricValue,
  formatPercent,
  formatRelativeTime,
  formatSizeMb,
  metricExceeded,
  metricLabel,
  metricLimitPhrase,
  pluralize,
  romanNumeral,
} from "./format";
import { describeEnv, notJudgedReason, summarizeRun } from "./run-summary";

describe("format", () => {
  it("formats durations across magnitudes", () => {
    expect(formatDuration(0.44)).toBe("0.4 ms");
    expect(formatDuration(12.34)).toBe("12.3 ms");
    expect(formatDuration(184.4)).toBe("184 ms");
    expect(formatDuration(1840.5)).toBe("1.84 s");
    expect(formatDuration(65_000)).toBe("1 min 5 s");
    expect(formatDuration(120_000)).toBe("2 min");
  });

  it("formats metrics and limits", () => {
    expect(formatMetricValue(36, "tokens")).toBe("36 tokens");
    expect(formatMetricValue(212.4, "x")).toBe("212.4×");
    expect(formatMetricValue(9204, "ms")).toBe("9,204 ms");
    expect(formatMetricValue(3)).toBe("3");
    expect(metricLabel("model_tokens")).toBe("Model tokens");
    expect(metricExceeded({ name: "a", value: 41, limit: 36 })).toBe(true);
    expect(metricExceeded({ name: "a", value: 36, limit: 36 })).toBe(false);
    expect(metricExceeded({ name: "a", value: 99 })).toBe(false);
  });

  it("reads a limit the right way round", () => {
    const throughput = { name: "throughput", unit: "tok/s", limit: 20, higherIsBetter: true };
    // Higher is better: the limit is a floor.
    expect(metricExceeded({ ...throughput, value: 41 })).toBe(false);
    expect(metricExceeded({ ...throughput, value: 20 })).toBe(false);
    expect(metricExceeded({ ...throughput, value: 12 })).toBe(true);
    expect(metricLimitPhrase({ ...throughput, value: 41 })).toBe("at least");
    expect(metricLimitPhrase({ ...throughput, value: 12 })).toBe("under the minimum of");
    // Lower is better: a ceiling, as before.
    expect(metricLimitPhrase({ name: "a", value: 36, limit: 36 })).toBe("limit");
    expect(metricLimitPhrase({ name: "a", value: 41, limit: 36 })).toBe("over the limit of");
  });

  it("formats sizes in decimal units", () => {
    expect(formatBytes(0)).toBe("0 kB");
    expect(formatBytes(412_000)).toBe("412 kB");
    expect(formatBytes(7_156_949)).toBe("7.2 MB");
    expect(formatBytes(711_966_212)).toBe("712 MB");
    expect(formatBytes(3_897_310_556)).toBe("3.90 GB");
    expect(formatBytes(-5)).toBe("0 kB");
    expect(formatSizeMb(712)).toBe("712 MB");
    expect(formatSizeMb(3900)).toBe("3.9 GB");
    expect(formatSizeMb(3.6)).toBe("3.6 MB");
    expect(formatSizeMb(0.01)).toBe("under 0.1 MB");
  });

  it("never shows 100% before a download is finished", () => {
    expect(formatPercent(0, 1000)).toBe("0%");
    expect(formatPercent(304, 1000)).toBe("30%");
    expect(formatPercent(999, 1000)).toBe("99%");
    expect(formatPercent(1000, 1000)).toBe("100%");
    expect(formatPercent(5, 0)).toBe("0%");
    expect(formatPercent(2000, 1000)).toBe("100%");
  });

  it("formats relative times against a fixed clock", () => {
    const now = new Date(2026, 10, 14, 15, 0, 0);
    const ago = (ms: number) => new Date(now.getTime() - ms).toISOString();
    expect(formatRelativeTime(ago(10_000), now)).toBe("just now");
    expect(formatRelativeTime(ago(12 * 60_000), now)).toBe("12 min ago");
    expect(formatRelativeTime(ago(3 * 3_600_000), now)).toBe("3 h ago");
    expect(formatRelativeTime(ago(20 * 3_600_000), now)).toBe("yesterday");
    expect(formatRelativeTime(ago(4 * 86_400_000), now)).toBe("4 days ago");
    expect(formatRelativeTime(ago(30 * 86_400_000), now)).toBe("Oct 15");
  });

  it("pluralizes", () => {
    expect(pluralize(1, "case")).toBe("1 case");
    expect(pluralize(1200, "problem")).toBe("1,200 problems");
  });
});

describe("run summary", () => {
  const base = { passed: 2, total: 7, durationMs: 84.2 };
  it("says what happened for each verdict", () => {
    expect(summarizeRun({ ...base, passed: 7, verdict: "accepted", mode: "submit" })).toBe(
      "All 7 cases passed in 84.2 ms.",
    );
    expect(summarizeRun({ ...base, passed: 3, total: 3, verdict: "accepted", mode: "run" })).toContain(
      "Submit to run the hidden cases",
    );
    expect(summarizeRun({ ...base, verdict: "wrong_answer", mode: "submit" })).toBe("2 of 7 cases passed.");
    expect(summarizeRun({ ...base, verdict: "time_limit", mode: "submit" })).toContain("time limit");
    expect(summarizeRun({ ...base, verdict: "canceled", mode: "run" })).toBe(
      "2 of 7 cases had passed when you canceled.",
    );
    expect(summarizeRun({ ...base, total: 0, passed: 0, verdict: "canceled", mode: "run" })).toBe(
      "Canceled before any case ran.",
    );
    expect(summarizeRun({ ...base, verdict: "harness_error", mode: "run" })).toContain("not with your code");
  });

  it("counts passed and skipped cases for a run that was not judged", () => {
    expect(
      summarizeRun({ ...base, passed: 2, total: 4, skipped: 2, verdict: "assets_missing", mode: "submit" }),
    ).toBe("2 of 4 cases passed, 2 skipped.");
    expect(
      summarizeRun({ ...base, passed: 2, total: 3, skipped: 1, verdict: "hardware_missing", mode: "run" }),
    ).toBe("2 of 3 cases passed, 1 skipped.");
    // Neither reason calls it a failure.
    expect(notJudgedReason("assets_missing")).toContain("Nothing that ran failed");
    expect(notJudgedReason("assets_missing")).toContain("not downloaded");
    expect(notJudgedReason("hardware_missing")).toContain("hardware");
  });

  it("describes the environment", () => {
    expect(describeEnv({ python: "3.15.2", torch: "2.15.0", device: "mps" })).toBe(
      "Python 3.15.2, torch 2.15.0, mps",
    );
    expect(describeEnv({})).toBe("");
  });
});

describe("romanNumeral", () => {
  it("numbers modules", () => {
    expect([1, 2, 3, 4, 5, 9, 10, 14, 15, 40].map(romanNumeral)).toEqual([
      "I",
      "II",
      "III",
      "IV",
      "V",
      "IX",
      "X",
      "XIV",
      "XV",
      "XL",
    ]);
  });
});
