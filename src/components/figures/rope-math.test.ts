import { describe, expect, it } from "vitest";
import { frequencies, pairProducts, rope, ropeScore } from "./rope-math";

const round = (values: number[], digits = 6) => values.map((value) => Number(value.toFixed(digits)) + 0);
/** The statements print float32 results to six decimals; float64 agrees to within one unit of the last. */
const expectStated = (values: number[], stated: number[]) => {
  expect(values).toHaveLength(stated.length);
  values.forEach((value, index) => {
    expect(Math.abs(value - (stated[index] ?? Number.NaN))).toBeLessThan(1e-6);
  });
};
const dot = (a: number[], b: number[]) => pairProducts(a, b).reduce((sum, part) => sum + part, 0);

describe("frequencies, as problems/transformer/rope states them", () => {
  it("are base^(-2i/D): 1 for the first pair, the slowest for the last", () => {
    expect(round(frequencies(4, 100))).toEqual([1, 0.1]);
    const thetas = frequencies(8, 10000);
    expect(thetas[0]).toBe(1);
    expect(thetas[3]).toBeCloseTo(10000 ** (-6 / 8), 12);
  });
});

describe("rope", () => {
  const x = [1.0, 0.0, 1.0, 0.0];

  it("reproduces the example of the rope statement", () => {
    expect(rope(x, 0, 100)).toEqual(x);
    expectStated(rope(x, 1, 100), [0.540302, 0.841471, 0.995004, 0.099833]);
  });

  it("reproduces the example of the cached-rope-offsets statement", () => {
    expectStated(rope(x, 3, 100), [-0.989992, 0.14112, 0.955337, 0.29552]);
    expectStated(rope(x, 4, 100), [-0.653644, -0.756802, 0.921061, 0.389418]);
  });

  it("turns counterclockwise", () => {
    // A quarter turn takes (1, 0) to (0, 1), not to (0, -1).
    expect(round(rope([1, 0], Math.PI / 2, 100))).toEqual([0, 1]);
  });

  it("pairs adjacent dimensions, not dimension i with i + D/2", () => {
    expect(round(rope([1, 0, 0, 0], 1, 100))).toEqual([0.540302, 0.841471, 0, 0]);
  });

  it("keeps the length of every pair", () => {
    const y = rope([0.3, -1.2, 0.8, 0.5], 37, 100);
    expect(pairProducts(y, y)[0]).toBeCloseTo(0.3 ** 2 + 1.2 ** 2, 12);
    expect(pairProducts(y, y)[1]).toBeCloseTo(0.8 ** 2 + 0.5 ** 2, 12);
  });
});

describe("ropeScore", () => {
  const unit = [1.0, 0.0, 1.0, 0.0];
  const q = [0.3, -1.2, 0.8, 0.5];
  const k = [-0.7, 0.4, 1.1, -0.2];

  it("gives the score of the cached-rope-offsets example in both rows", () => {
    // The second new query against the first new key: positions 4 and 3 in row 0, 1 and 0 in row 1.
    const row0 = ropeScore(unit, unit, 4, 3, 100);
    const row1 = ropeScore(unit, unit, 1, 0, 100);
    expect(row0.score).toBeCloseTo(Math.cos(1) + Math.cos(0.1), 12);
    expect(Number(row0.score.toFixed(4))).toBe(1.5353);
    expect(Number(row1.score.toFixed(4))).toBe(1.5353);
    expect(round(row1.pairs, 4)).toEqual([0.5403, 0.995]);
    expect(row0.q).not.toEqual(row1.q);
  });

  it("does not change when one shift is added to both positions", () => {
    for (const [m, n] of [
      [0, 0],
      [2, 5],
      [7, 1],
      [8, 8],
    ] as const) {
      const unshifted = ropeScore(q, k, m, n, 100);
      for (const shift of [1, 3, 17, 37, 40]) {
        const shifted = ropeScore(q, k, m + shift, n + shift, 100);
        expect(shifted.score).toBeCloseTo(unshifted.score, 10);
        shifted.pairs.forEach((part, pair) => {
          expect(part).toBeCloseTo(unshifted.pairs[pair] ?? Number.NaN, 10);
        });
      }
    }
  });

  it("changes with the difference of the positions", () => {
    expect(ropeScore(q, k, 2, 5, 100).score).not.toBeCloseTo(ropeScore(q, k, 2, 6, 100).score, 3);
    expect(ropeScore(unit, unit, 3, 3, 100).score).toBeCloseTo(2, 12);
  });

  it("equals the score of the plain query against the key rotated by n - m", () => {
    for (const [m, n] of [
      [37, 38],
      [5, 2],
      [0, 9],
    ] as const) {
      expect(ropeScore(q, k, m, n, 100).score).toBeCloseTo(dot(q, rope(k, n - m, 100)), 10);
    }
  });
});
