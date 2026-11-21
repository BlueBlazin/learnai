import { describe, expect, it } from "vitest";
import {
  acceptanceRate,
  acceptProbability,
  bump,
  emitted,
  overlap,
  residual,
  residualTotal,
  speculativeStep,
  TARGET,
  VOCABULARY,
} from "./draft-math";

const round = (values: readonly number[], digits = 6) =>
  values.map((value) => Number(value.toFixed(digits)) + 0);
const total = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0);

/** The example of problems/speculative/speculative-sampling-step. */
const P = [0.1, 0.4, 0.2, 0.3];
const Q = [0.5, 0.2, 0.2, 0.1];

describe("the example of the speculative-sampling-step statement", () => {
  it("has the residual [0.0, 0.2, 0.0, 0.2] and R = 0.4", () => {
    expect(round(residual(P, Q))).toEqual([0, 0.2, 0, 0.2]);
    expect(residualTotal(P, Q)).toBeCloseTo(0.4, 12);
  });

  it("accepts token 0 with probability 0.1 / 0.5 = 0.2 and token 1 always", () => {
    expect(acceptProbability(P, Q, 0)).toBeCloseTo(0.2, 12);
    expect(acceptProbability(P, Q, 1)).toBe(1);
  });

  it("gives the four calls of the example", () => {
    expect(speculativeStep(P, Q, 0, 0.1, 0.9)).toEqual({ token: 0, accepted: true });
    expect(speculativeStep(P, Q, 0, 0.6, 0.3)).toEqual({ token: 1, accepted: false });
    expect(speculativeStep(P, Q, 0, 0.6, 0.7)).toEqual({ token: 3, accepted: false });
    expect(speculativeStep(P, Q, 1, 0.99, 0.5)).toEqual({ token: 1, accepted: true });
  });

  it("emits token 1 with probability 0.2 + 0.4 * 0.5 = 0.4 = p[1]", () => {
    // Only token 0 can be rejected, with probability 0.5 * 0.8 = 0.4.
    expect(Q[0] * (1 - acceptProbability(P, Q, 0))).toBeCloseTo(0.4, 12);
    expect(emitted(P, Q)[1]).toBeCloseTo(0.4, 12);
  });

  it("has the acceptance rate sum of min(p, q) = 0.6", () => {
    expect(round(overlap(P, Q))).toEqual([0.1, 0.2, 0.2, 0.1]);
    expect(acceptanceRate(P, Q)).toBeCloseTo(0.6, 12);
  });
});

describe("speculativeStep", () => {
  it("accepts for every u_accept when p[x] >= q[x]", () => {
    for (const u of [0, 0.5, 0.999999]) expect(speculativeStep(P, Q, 3, u, 0).accepted).toBe(true);
  });

  it("rejects for every u_accept, also 0, when p[x] = 0", () => {
    const step = speculativeStep([0, 1], [0.5, 0.5], 0, 0, 0);
    expect(step).toEqual({ token: 1, accepted: false });
  });

  it("resamples the first token with a positive residual at u_resample = 0", () => {
    expect(speculativeStep(P, Q, 0, 0.9, 0)).toEqual({ token: 1, accepted: false });
  });

  it("never resamples the draft token or a token with p <= q", () => {
    for (let step = 0; step < 100; step++) {
      const { token } = speculativeStep(P, Q, 0, 0.9, step / 100);
      expect([1, 3]).toContain(token);
    }
  });
});

describe("accepting and resampling together", () => {
  const pairs: [readonly number[], readonly number[]][] = [
    [P, Q],
    [Q, P],
    [P, P],
    [
      [0, 0.5, 0.5],
      [0.6, 0.4, 0],
    ],
    [TARGET, bump(2, 1)],
    [TARGET, bump(0.5, -2)],
    [TARGET, bump(3, 2)],
  ];

  it("give back p", () => {
    for (const [p, q] of pairs) {
      emitted(p, q).forEach((probability, token) => {
        expect(probability).toBeCloseTo(p[token] ?? Number.NaN, 12);
      });
    }
  });

  it("split p into the overlap and the residual, token by token", () => {
    for (const [p, q] of pairs) {
      const shared = overlap(p, q);
      residual(p, q).forEach((rest, token) => {
        expect((shared[token] ?? 0) + rest).toBeCloseTo(p[token] ?? Number.NaN, 12);
      });
      expect(acceptanceRate(p, q) + residualTotal(p, q)).toBeCloseTo(1, 12);
    }
  });

  it("match the frequencies of the step on a grid of the two uniform numbers", () => {
    // Every draft token with a 200 by 200 grid of midpoints for (u_accept, u_resample).
    const counts = new Array<number>(P.length).fill(0);
    let accepted = 0;
    const side = 200;
    Q.forEach((draft, x) => {
      for (let a = 0; a < side; a++) {
        for (let b = 0; b < side; b++) {
          const step = speculativeStep(P, Q, x, (a + 0.5) / side, (b + 0.5) / side);
          counts[step.token] = (counts[step.token] ?? 0) + draft / side ** 2;
          if (step.accepted) accepted += draft / side ** 2;
        }
      }
    });
    counts.forEach((frequency, token) => {
      expect(frequency).toBeCloseTo(P[token] ?? Number.NaN, 2);
    });
    expect(accepted).toBeCloseTo(0.6, 2);
  });
});

describe("the distributions of the figure", () => {
  it("are distributions over the vocabulary", () => {
    for (const q of [TARGET, bump(0.5, -2), bump(3, 2), bump(1.5, 0.25)]) {
      expect(q).toHaveLength(VOCABULARY);
      expect(total(q)).toBeCloseTo(1, 12);
      for (const probability of q) expect(probability).toBeGreaterThan(0);
    }
  });

  it("make the draft equal to the target at sharpness 1 and shift 0", () => {
    expect(bump(1, 0)).toEqual([...TARGET]);
    expect(acceptanceRate(TARGET, bump(1, 0))).toBeCloseTo(1, 12);
  });

  it("lose acceptance as the draft moves away or changes its sharpness", () => {
    const rate = (sharpness: number, shift: number) => acceptanceRate(TARGET, bump(sharpness, shift));
    expect(rate(1, 1)).toBeLessThan(rate(1, 0.5));
    expect(rate(1, 2)).toBeLessThan(rate(1, 1));
    expect(rate(2, 0)).toBeLessThan(1);
    expect(rate(0.5, 0)).toBeLessThan(1);
    expect(rate(3, 0)).toBeLessThan(rate(2, 0));
  });

  it("stay under the top of the figure's axis", () => {
    for (let sharpness = 0.5; sharpness <= 3; sharpness += 0.25) {
      for (let shift = -2; shift <= 2; shift += 0.25) {
        expect(Math.max(...bump(sharpness, shift))).toBeLessThan(0.5);
      }
    }
  });
});
