/**
 * One verification step of speculative sampling behind the `draft-overlap`
 * figure, exactly as problems/speculative/speculative-sampling-step states it:
 * a draft token `x` sampled from `q` is accepted with probability
 * `min(1, p[x] / q[x])`, and a rejected one is replaced by a sample from the
 * normalized residual `r / R`, where `r_i = max(0, p_i - q_i)` and `R` is its
 * total.
 */

const sum = (values: readonly number[]) => values.reduce((total, value) => total + value, 0);

/** The overlap of the two distributions, token by token: `min(p_i, q_i)`. */
export function overlap(p: readonly number[], q: readonly number[]): number[] {
  return p.map((target, token) => Math.min(target, q[token] ?? 0));
}

/** The residual `r_i = max(0, p_i - q_i)`, not normalized. */
export function residual(p: readonly number[], q: readonly number[]): number[] {
  return p.map((target, token) => Math.max(0, target - (q[token] ?? 0)));
}

/** `R`: the total of the residual, which is also the probability that a draft token is rejected. */
export function residualTotal(p: readonly number[], q: readonly number[]): number {
  return sum(residual(p, q));
}

/** The probability that the draft token `x` is accepted: `min(1, p[x] / q[x])`, with `q[x] > 0`. */
export function acceptProbability(p: readonly number[], q: readonly number[], x: number): number {
  return Math.min(1, (p[x] ?? 0) / (q[x] ?? 0));
}

/** The probability that a token drawn from `q` is accepted: the sum of `min(p_i, q_i)`. */
export function acceptanceRate(p: readonly number[], q: readonly number[]): number {
  return sum(overlap(p, q));
}

/**
 * The step itself, with the two uniform numbers supplied by the caller.
 * Accept if and only if `uAccept * q[x] < p[x]`. Otherwise the token is the
 * first `k` whose cumulative residual `c_k` is above `uResample * R`.
 */
export function speculativeStep(
  p: readonly number[],
  q: readonly number[],
  x: number,
  uAccept: number,
  uResample: number,
): { token: number; accepted: boolean } {
  if (uAccept * (q[x] ?? 0) < (p[x] ?? 0)) return { token: x, accepted: true };
  const r = residual(p, q);
  const threshold = uResample * sum(r);
  let cumulative = 0;
  for (let token = 0; token < r.length; token++) {
    cumulative += r[token] ?? 0;
    if (cumulative > threshold) return { token, accepted: false };
  }
  // Not reached for a rejected token: a rejection implies R > 0 and uResample < 1.
  return { token: r.length - 1, accepted: false };
}

/**
 * The distribution of the emitted token when `x` is drawn from `q`: token `i`
 * is drafted and accepted, or some draft is rejected and resampling picks it.
 * It is computed from the two branches of the step; that it equals `p` is the
 * guarantee.
 */
export function emitted(p: readonly number[], q: readonly number[]): number[] {
  const r = residual(p, q);
  const total = sum(r);
  // A token that `q` never drafts is never accepted or rejected.
  const kept = q.map((draft, token) => (draft > 0 ? draft * acceptProbability(p, q, token) : 0));
  const rejected = 1 - sum(kept);
  return kept.map((mass, token) => mass + (total > 0 ? (rejected * (r[token] ?? 0)) / total : 0));
}

/** The vocabulary of the figure and the bump its two distributions are cut from. */
export const VOCABULARY = 8;
const CENTER = 3;
const WIDTH = 1.5;

/**
 * A distribution over the figure's vocabulary: a bump around token
 * `CENTER + shift` whose logits are multiplied by `sharpness`. Sharpness 1 and
 * shift 0 give the target; a higher sharpness narrows the bump.
 */
export function bump(sharpness: number, shift: number): number[] {
  const weights = Array.from({ length: VOCABULARY }, (_, token) =>
    Math.exp((-sharpness * (token - CENTER - shift) ** 2) / (2 * WIDTH ** 2)),
  );
  const total = sum(weights);
  return weights.map((weight) => weight / total);
}

/** The target distribution `p` of the figure. */
export const TARGET: readonly number[] = bump(1, 0);
