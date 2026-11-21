/**
 * The two quantizers behind the `quant-grid` figure, each exactly as its
 * problem statement defines it: the affine grid of
 * problems/quantization/affine-int8 and the symmetric clipped grid of
 * problems/quantization/clipping-calibration. Both take the number of bits, so
 * the affine grid at 8 bits is the int8 grid of its statement.
 *
 * The clipping fraction `c` is the one of clipping-calibration: the grid
 * reaches `c` times the full range, and 1 means no clipping. The affine
 * statement has no clipping; here its range `[lo, hi]` is multiplied by `c`,
 * which keeps 0.0 on the grid and is the statement's own grid at `c = 1`.
 */

export type GridMode = "symmetric" | "affine";

/** A grid of codes: the real value of code `q` is `scale * (q - zeroPoint)`. */
export interface Grid {
  scale: number;
  zeroPoint: number;
  /** The lowest and highest code. */
  qmin: number;
  qmax: number;
  /** The range the grid was fitted to. A value beyond it is clipped: it saturates at the last code. */
  low: number;
  high: number;
}

/** Rounding to the nearest integer, ties to the even one, as `round` does in PyTorch, JAX, and MLX. */
export function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest < 0.5) return floor;
  if (rest > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

const clamp = (value: number, low: number, high: number) => Math.min(Math.max(value, low), high);

/**
 * The affine grid of `quant_params`: the range is widened to include 0, `lo`
 * maps to the lowest code, `hi` to the highest, and 0.0 to `zeroPoint`
 * exactly. Rounding the zero point moves the levels by up to half a step, so
 * the last levels need not sit exactly on `lo` and `hi`. At 8 bits the codes are -128 to 127 and the scale is
 * `(hi - lo) / 255`. A range of zero width gets scale 1.
 */
export function affineGrid(samples: readonly number[], bits = 8, clip = 1): Grid {
  const qmin = -(2 ** (bits - 1));
  const qmax = 2 ** (bits - 1) - 1;
  const lo = clip * Math.min(...samples, 0);
  const hi = clip * Math.max(...samples, 0);
  const scale = hi > lo ? (hi - lo) / (qmax - qmin) : 1;
  const zeroPoint = clamp(qmin - roundHalfEven(lo / scale), qmin, qmax);
  return { scale, zeroPoint, qmin, qmax, low: lo, high: hi };
}

/**
 * The symmetric grid of `best_clip` for one candidate fraction: with `m` the
 * largest magnitude and `qmax = 2^(bits - 1) - 1`, the limit is `clip * m`,
 * the scale is `limit / qmax`, and the codes are `-qmax` to `qmax`.
 */
export function symmetricGrid(samples: readonly number[], bits: number, clip: number): Grid {
  const qmax = 2 ** (bits - 1) - 1;
  const limit = clip * Math.max(...samples.map(Math.abs));
  return { scale: limit / qmax, zeroPoint: 0, qmin: -qmax, qmax, low: -limit, high: limit };
}

export function makeGrid(mode: GridMode, samples: readonly number[], bits: number, clip: number): Grid {
  return mode === "affine" ? affineGrid(samples, bits, clip) : symmetricGrid(samples, bits, clip);
}

/** The code of `x`: rounded, moved by the zero point, and clamped, so a value beyond the grid saturates. */
export function quantize(x: number, grid: Grid): number {
  return clamp(roundHalfEven(x / grid.scale) + grid.zeroPoint, grid.qmin, grid.qmax);
}

/** The real value of code `q`. */
export function dequantize(q: number, grid: Grid): number {
  return grid.scale * (q - grid.zeroPoint);
}

/** The real value of every code, lowest first. */
export function gridLevels(grid: Grid): number[] {
  return Array.from({ length: grid.qmax - grid.qmin + 1 }, (_, index) => dequantize(grid.qmin + index, grid));
}

/**
 * Whether `x` is clipped: it lies beyond the range the grid was fitted to. In
 * the words of both statements such a value saturates, and every other value
 * is rounded to the nearest level.
 */
export function isClipped(x: number, grid: Grid): boolean {
  return x < grid.low || x > grid.high;
}

export interface QuantError {
  /** The mean squared error over all samples: `clipping + rounding`. */
  mse: number;
  /** The part of `mse` from clipped samples, which saturate at the last code. */
  clipping: number;
  /** The part of `mse` from the other samples, which are rounded to the nearest level. */
  rounding: number;
  /** How many samples are clipped. */
  clipped: number;
}

/**
 * The mean squared error of quantizing `samples` on `grid` and mapping the
 * codes back, split by whether each sample is clipped. Both parts are divided
 * by the number of all samples, so they add up to the MSE of
 * clipping-calibration.
 */
export function quantError(samples: readonly number[], grid: Grid): QuantError {
  let clipping = 0;
  let rounding = 0;
  let clipped = 0;
  for (const x of samples) {
    const squared = (dequantize(quantize(x, grid), grid) - x) ** 2;
    if (isClipped(x, grid)) {
      clipping += squared;
      clipped += 1;
    } else {
      rounding += squared;
    }
  }
  const count = samples.length;
  return {
    mse: (clipping + rounding) / count,
    clipping: clipping / count,
    rounding: rounding / count,
    clipped,
  };
}

/** A generator of uniform numbers in [0, 1) from a seed (mulberry32), so the sample is the same on every run. */
function uniformFrom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = Math.imul(state ^ (state >>> 15), state | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The fixed sample the figure quantizes: `count` weights shaped like a real
 * layer's, a narrow bell a little above zero with a few values far out, more
 * of them on the positive side. Nine in ten come from the bell and the rest
 * from a wide one; every value is rounded to three decimals.
 */
export function sampleWeights(count = 600, seed = 11): number[] {
  const uniform = uniformFrom(seed);
  const normal = () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
  return Array.from({ length: count }, () => {
    const wide = uniform() < 0.1;
    const value = wide ? 0.35 + 0.6 * normal() : 0.12 + 0.17 * normal();
    return Math.round(value * 1000) / 1000;
  });
}

/** Counts of `samples` in `bins` equal bins over `[low, high]`; the top edge belongs to the last bin. */
export function histogram(samples: readonly number[], low: number, high: number, bins: number): number[] {
  const counts = new Array<number>(bins).fill(0);
  const width = (high - low) / bins;
  for (const x of samples) {
    const bin = clamp(Math.floor((x - low) / width), 0, bins - 1);
    counts[bin] = (counts[bin] ?? 0) + 1;
  }
  return counts;
}
