/**
 * RoPE behind the `rope-shift` figure, exactly as problems/transformer/rope
 * states it (problems/kv-cache/cached-rope-offsets uses the same function):
 * a vector of even length `D` is split into adjacent pairs
 * `(x[2i], x[2i + 1])`, pair `i` has the frequency `base^(-2i / D)`, and it is
 * rotated counterclockwise by the position times that frequency.
 */

/** The frequency of every pair: `base^(-2i / D)` for `i = 0, ..., D/2 - 1`. The first is 1. */
export function frequencies(dimension: number, base: number): number[] {
  return Array.from({ length: dimension / 2 }, (_, pair) => base ** ((-2 * pair) / dimension));
}

/** `x` rotated for `position`: each adjacent pair turned counterclockwise by `position * theta_i`. */
export function rope(x: readonly number[], position: number, base: number): number[] {
  const out = new Array<number>(x.length).fill(0);
  frequencies(x.length, base).forEach((theta, pair) => {
    const cos = Math.cos(position * theta);
    const sin = Math.sin(position * theta);
    const first = x[2 * pair] ?? 0;
    const second = x[2 * pair + 1] ?? 0;
    out[2 * pair] = first * cos - second * sin;
    out[2 * pair + 1] = first * sin + second * cos;
  });
  return out;
}

/** What each pair adds to the dot product of `a` and `b`; the dot product is their sum. */
export function pairProducts(a: readonly number[], b: readonly number[]): number[] {
  return Array.from(
    { length: a.length / 2 },
    (_, pair) => (a[2 * pair] ?? 0) * (b[2 * pair] ?? 0) + (a[2 * pair + 1] ?? 0) * (b[2 * pair + 1] ?? 0),
  );
}

export interface RopeScore {
  /** The query and the key after their rotations. */
  q: number[];
  k: number[];
  /** The part of the score from each pair. */
  pairs: number[];
  /** The attention score: the dot product of the rotated query and the rotated key. */
  score: number;
}

/**
 * The score of a query at `queryPosition` against a key at `keyPosition`:
 * both are rotated for their own positions and then multiplied. Nothing here
 * uses the difference of the positions; that the score depends on it alone is
 * what the figure shows.
 */
export function ropeScore(
  q: readonly number[],
  k: readonly number[],
  queryPosition: number,
  keyPosition: number,
  base: number,
): RopeScore {
  const rotatedQ = rope(q, queryPosition, base);
  const rotatedK = rope(k, keyPosition, base);
  const pairs = pairProducts(rotatedQ, rotatedK);
  return { q: rotatedQ, k: rotatedK, pairs, score: pairs.reduce((sum, part) => sum + part, 0) };
}
