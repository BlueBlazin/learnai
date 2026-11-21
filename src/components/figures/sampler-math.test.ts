import { describe, expect, it } from "vitest";
import { applyTemperature, keptCount, runSampler, softmax, topKFilter, topPFilter } from "./sampler-math";

const NEG = Number.NEGATIVE_INFINITY;
const kept = (logits: number[]) => logits.flatMap((logit, token) => (logit === NEG ? [] : [token]));
const round = (values: number[], digits = 3) => values.map((value) => Number(value.toFixed(digits)));

describe("softmax", () => {
  it("sums to 1 and gives removed tokens probability 0", () => {
    const probabilities = softmax([0, Math.log(4), NEG, Math.log(3)]);
    expect(probabilities[2]).toBe(0);
    expect(probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    expect(round(softmax([0, Math.log(4), Math.log(2), Math.log(3)]), 2)).toEqual([0.1, 0.4, 0.2, 0.3]);
  });
});

describe("topKFilter, as problems/sampling/top-k-filtering states it", () => {
  it("reproduces the statement's example, ties by lowest index", () => {
    expect(topKFilter([1.0, 3.0, 0.5, 2.0, -1.0], 2)).toEqual([NEG, 3.0, NEG, 2.0, NEG]);
    expect(topKFilter([0.0, 2.0, NEG, 2.0, 2.0], 2)).toEqual([NEG, 2.0, NEG, 2.0, NEG]);
    expect(kept(topKFilter([0.0, 2.0, NEG, 2.0, 2.0], 4))).toEqual([0, 1, 3, 4]);
  });

  it("is switched off by k = 0 and by k of at least V", () => {
    const row = [0.0, 2.0, NEG, 2.0, 2.0];
    for (const k of [0, 5, 100]) expect(topKFilter(row, k)).toEqual(row);
  });

  it("keeps exactly k of a row of equal logits", () => {
    expect(kept(topKFilter([1, 1, 1, 1], 3))).toEqual([0, 1, 2]);
  });
});

describe("topPFilter, as problems/sampling/top-p-filtering states it", () => {
  it("keeps the token that carries the mass to or past p", () => {
    // softmax = [0.1, 0.4, 0.2, 0.3]; masses 0.4, 0.7: token 3 crosses 0.6 and is kept.
    const row = topPFilter([0.0, Math.log(4), Math.log(2), Math.log(3)], 0.6);
    expect(kept(row)).toEqual([1, 3]);
    expect(round(softmax(row), 4)).toEqual([0, 0.5714, 0, 0.4286]);
    // softmax = [0.7, 0.1, 0.1, 0.1]: the top token alone reaches 0.6.
    expect(kept(topPFilter([Math.log(7), 0.0, 0.0, 0.0], 0.6))).toEqual([0]);
  });

  it("ranks equal probabilities by index", () => {
    expect(kept(topPFilter([0, 0, 0, 0], 0.6))).toEqual([0, 1, 2]);
  });

  it("always keeps the top token, and removes nothing at p = 1", () => {
    expect(kept(topPFilter([0, 1, 2, 3], 0.0001))).toEqual([3]);
    const row = [5, -20, NEG, 0];
    expect(topPFilter(row, 1)).toEqual(row);
  });

  it("works on what an earlier stage left", () => {
    // The sampler-chain example after top-k 4: [0.135, 0.473, 0.287, 0.105, 0, 0].
    const row = topPFilter([1.25, 2.5, 2.0, 1.0, NEG, NEG], 0.8);
    expect(kept(row)).toEqual([0, 1, 2]);
    expect(round(softmax(row))).toEqual([0.151, 0.528, 0.32, 0, 0, 0]);
  });
});

describe("runSampler, in the two orders of problems/sampling/sampler-chain", () => {
  // The statement's example after the penalties; min-p is not part of the figure.
  const penalized = [1.25, 2.5, 2.0, 1.0, 0.0, -1.95];
  const settings = { topK: 4, topP: 0.8, temperature: 2.0 };

  it("divides by the temperature without bringing removed tokens back", () => {
    expect(applyTemperature([1.25, 2.5, NEG], 2)).toEqual([0.625, 1.25, NEG]);
  });

  it("with the filters first, cuts the model's own distribution and then flattens it", () => {
    const result = runSampler(penalized, { ...settings, order: "filters-first" });
    expect(result.removedBy).toEqual([null, null, null, "top-p", "top-k", "top-k"]);
    expect(keptCount(result)).toBe(3);
    expect(result.probabilities).toEqual(softmax([0.625, 1.25, 1.0, NEG, NEG, NEG]));
  });

  it("with the temperature first, filters the flattened distribution", () => {
    // The statement gives this distribution to three decimals.
    const flattened = softmax(applyTemperature(penalized, 2));
    [0.168, 0.314, 0.245, 0.149, 0.09, 0.034].forEach((stated, token) => {
      expect(Math.abs((flattened[token] ?? 0) - stated)).toBeLessThan(1e-3);
    });
    const result = runSampler(penalized, { ...settings, order: "temperature-first" });
    expect(result.removedBy).toEqual([null, null, null, "top-p", "top-k", "top-k"]);
  });

  it("keeps a different set in the other order for the figure's opening settings", () => {
    const logits = [3.2, 2.9, 2.5, 2.2, 1.8, 1.4, 0.9, 0.4, -0.2, -0.9];
    const opening = { topK: 8, topP: 0.9, temperature: 2 };
    const filtersFirst = runSampler(logits, { ...opening, order: "filters-first" });
    const temperatureFirst = runSampler(logits, { ...opening, order: "temperature-first" });
    expect(keptCount(temperatureFirst)).toBeGreaterThan(keptCount(filtersFirst));
    for (const result of [filtersFirst, temperatureFirst]) {
      expect(result.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      result.removedBy.forEach((stage, token) => {
        expect(result.probabilities[token] === 0).toBe(stage !== null);
      });
    }
  });

  it("removes nothing when both filters are off", () => {
    const result = runSampler(penalized, { topK: 6, topP: 1, temperature: 0.5, order: "filters-first" });
    expect(keptCount(result)).toBe(6);
  });
});
