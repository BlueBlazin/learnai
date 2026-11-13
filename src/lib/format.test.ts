import { describe, expect, it } from "vitest";
import {
  formatDuration,
  formatMetricValue,
  formatRelativeTime,
  metricExceeded,
  metricLabel,
  pluralize,
} from "./format";
import { describeEnv, summarizeRun } from "./run-summary";

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

  it("describes the environment", () => {
    expect(describeEnv({ python: "3.15.2", torch: "2.15.0", device: "mps" })).toBe(
      "Python 3.15.2, torch 2.15.0, mps",
    );
    expect(describeEnv({})).toBe("");
  });
});
