import { describe, expect, it } from "vitest";
import {
  affineGrid,
  dequantize,
  gridLevels,
  histogram,
  isClipped,
  makeGrid,
  quantError,
  quantize,
  roundHalfEven,
  sampleWeights,
  symmetricGrid,
} from "./quant-math";

/** Rounded to `digits` decimals; adding 0 turns a negative zero into zero. */
const round = (values: number[], digits = 0) => values.map((value) => Number(value.toFixed(digits)) + 0);

describe("roundHalfEven", () => {
  it("rounds to the nearest integer and sends ties to the even one", () => {
    expect(round([30.6, 122.4, -51.0, -0.4, -1.6].map(roundHalfEven))).toEqual([31, 122, -51, 0, -2]);
    expect(round([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5].map(roundHalfEven))).toEqual([0, 2, 2, 4, 0, -2, -2]);
  });
});

describe("affineGrid, as problems/quantization/affine-int8 states it", () => {
  const x = [-0.5, 0.0, 0.3, 1.2, 2.0];

  it("reproduces the statement's example", () => {
    const grid = affineGrid(x);
    expect(grid.scale).toBeCloseTo(0.00980392, 8);
    expect(grid).toMatchObject({ zeroPoint: -77, qmin: -128, qmax: 127 });
    const codes = x.map((value) => quantize(value, grid));
    expect(codes).toEqual([-128, -77, -46, 45, 127]);
    expect(
      round(
        codes.map((code) => dequantize(code, grid)),
        6,
      ),
    ).toEqual([-0.5, 0, 0.303922, 1.196078, 2]);
    expect(dequantize(quantize(0, grid), grid)).toBe(0);
  });

  it("widens an all-positive range to include 0", () => {
    const grid = affineGrid([2.0, 3.0, 5.0]);
    expect(grid.scale).toBeCloseTo(0.0196078, 7);
    expect(grid.zeroPoint).toBe(-128);
    expect([2.0, 3.0, 5.0].map((value) => quantize(value, grid))).toEqual([-26, 25, 127]);
  });

  it("widens an all-negative range and gives an all-zero tensor scale 1", () => {
    expect(affineGrid([-4, -1]).zeroPoint).toBe(127);
    expect(affineGrid([0, 0, 0])).toMatchObject({ scale: 1, zeroPoint: -128 });
  });

  it("saturates values beyond the grid", () => {
    const grid = affineGrid(x);
    expect([-9, 9].map((value) => quantize(value, grid))).toEqual([-128, 127]);
  });

  it("uses every code of the given width and keeps 0.0 on the grid when clipped", () => {
    for (const bits of [2, 3, 4, 8]) {
      for (const clip of [1, 0.6, 0.25]) {
        const grid = affineGrid(x, bits, clip);
        expect(grid.qmax - grid.qmin + 1).toBe(2 ** bits);
        expect(gridLevels(grid).some((level) => level === 0)).toBe(true);
      }
    }
    // The range is [-0.5c, 2c]: at c = 0.5 it is a quarter to one, 2 bits give four levels.
    expect(round(gridLevels(affineGrid(x, 2, 0.5)), 6)).toEqual([-0.416667, 0, 0.416667, 0.833333]);
  });

  it("fits the range of the samples at fraction 1 and clips nothing", () => {
    const samples = sampleWeights();
    for (const bits of [2, 3, 4, 5, 6, 7, 8]) {
      const grid = affineGrid(samples, bits, 1);
      expect(grid).toMatchObject({ low: -0.991, high: 2.072 });
      expect(quantError(samples, grid).clipped).toBe(0);
      // The lowest and highest sample get the first and the last code.
      expect([grid.low, grid.high].map((x) => quantize(x, grid))).toEqual([grid.qmin, grid.qmax]);
    }
  });
});

describe("symmetricGrid, as problems/quantization/clipping-calibration states it", () => {
  const samples = [0.8, -0.9, 1.1, -1.2, 0.6, 2.0];
  const quantized = (clip: number) => {
    const grid = symmetricGrid(samples, 2, clip);
    return round(
      samples.map((x) => dequantize(quantize(x, grid), grid)),
      6,
    );
  };

  it("has the statement's codes: -7 to 7 at 4 bits, three points at 2 bits", () => {
    expect(symmetricGrid(samples, 4, 1)).toMatchObject({ qmin: -7, qmax: 7, zeroPoint: 0 });
    expect(round(gridLevels(symmetricGrid(samples, 2, 0.5)), 12)).toEqual([-1, 0, 1]);
    expect(symmetricGrid(samples, 4, 0.25)).toMatchObject({ low: -0.5, high: 0.5 });
  });

  it("reproduces the three rows of the statement's example", () => {
    expect(quantized(1.0)).toEqual([0, 0, 2, -2, 0, 2]);
    expect(quantized(0.5)).toEqual([1, -1, 1, -1, 1, 1]);
    expect(quantized(0.25)).toEqual([0.5, -0.5, 0.5, -0.5, 0.5, 0.5]);
    const mse = (clip: number) => quantError(samples, symmetricGrid(samples, 2, clip)).mse;
    expect(mse(1.0)).toBeCloseTo(0.5433, 4);
    expect(mse(0.5)).toBeCloseTo(0.21, 10);
    expect(mse(0.25)).toBeCloseTo(0.56, 10);
  });

  it("splits the error of the example into clipping and rounding", () => {
    // c = 1: nothing lies beyond the limit, so the whole error is rounding.
    expect(quantError(samples, symmetricGrid(samples, 2, 1))).toMatchObject({ clipping: 0, clipped: 0 });
    // c = 0.5: 1.1, -1.2, and 2.0 lie beyond the limit 1.0, with squared errors 0.01, 0.04, and 1.
    const half = quantError(samples, symmetricGrid(samples, 2, 0.5));
    expect(half.clipped).toBe(3);
    expect(half.clipping).toBeCloseTo(1.05 / 6, 12);
    expect(half.rounding).toBeCloseTo(0.21 / 6, 12);
    // c = 0.25: every sample lies beyond the limit 0.5.
    const quarter = quantError(samples, symmetricGrid(samples, 2, 0.25));
    expect(quarter).toMatchObject({ clipped: 6, rounding: 0 });
    expect(quarter.clipping).toBeCloseTo(0.56, 12);
  });
});

describe("quantError on the figure's sample", () => {
  const samples = sampleWeights();

  it("is the same sample on every run", () => {
    expect(samples).toHaveLength(600);
    expect(samples).toEqual(sampleWeights());
    expect([Math.min(...samples), Math.max(...samples)]).toEqual([-0.991, 2.072]);
    expect(histogram(samples, -2.75, 2.75, 88).reduce((a, b) => a + b, 0)).toBe(600);
  });

  it("adds up to the mean squared error in both modes", () => {
    for (const mode of ["symmetric", "affine"] as const) {
      for (const bits of [2, 4, 8]) {
        for (const clip of [1, 0.6, 0.2]) {
          const grid = makeGrid(mode, samples, bits, clip);
          const error = quantError(samples, grid);
          const direct =
            samples.reduce((sum, x) => sum + (dequantize(quantize(x, grid), grid) - x) ** 2, 0) /
            samples.length;
          expect(error.clipping + error.rounding).toBeCloseTo(direct, 12);
          expect(error.mse).toBeCloseTo(direct, 12);
          expect(error.clipped).toBe(samples.filter((x) => isClipped(x, grid)).length);
        }
      }
    }
  });

  it("trades clipping error for rounding error as the range shrinks", () => {
    const at = (clip: number) => quantError(samples, symmetricGrid(samples, 4, clip));
    expect(at(1).clipping).toBe(0);
    expect(at(0.5).clipping).toBeGreaterThan(0);
    expect(at(0.5).rounding).toBeLessThan(at(1).rounding);
    // At 4 bits some clipping beats none, and too much is worse again.
    expect(at(0.7).mse).toBeLessThan(at(1).mse);
    expect(at(0.2).mse).toBeGreaterThan(at(0.7).mse);
  });

  it("keeps every rounded sample within half a step", () => {
    const grid = makeGrid("symmetric", samples, 4, 0.6);
    for (const x of samples) {
      if (!isClipped(x, grid)) {
        expect(Math.abs(dequantize(quantize(x, grid), grid) - x)).toBeLessThanOrEqual(grid.scale / 2 + 1e-12);
      }
    }
  });
});
