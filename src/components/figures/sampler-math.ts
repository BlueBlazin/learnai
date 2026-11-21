/**
 * The sampler stages behind the `sampler-order` figure, on one row of logits.
 * Each follows its problem statement exactly (problems/sampling/top-k-filtering,
 * top-p-filtering, sampler-chain): "removing" a token sets its logit to
 * -Infinity, and equal values are ranked by token index, lowest first.
 */

export type SamplerStage = "top-k" | "top-p";

/** `filters-first` is the order of llama.cpp and of the sampler-chain problem. */
export type StageOrder = "filters-first" | "temperature-first";

/** Softmax of one row. Removed tokens (-Infinity) get probability 0. */
export function softmax(logits: readonly number[]): number[] {
  const peak = Math.max(...logits);
  const weights = logits.map((logit) => (logit === Number.NEGATIVE_INFINITY ? 0 : Math.exp(logit - peak)));
  const total = weights.reduce((sum, weight) => sum + weight, 0);
  return weights.map((weight) => weight / total);
}

/** Token indices by decreasing value; equal values by index, lowest first. */
function ranking(values: readonly number[]): number[] {
  return values.map((_, index) => index).sort((a, b) => (values[b] ?? 0) - (values[a] ?? 0) || a - b);
}

/**
 * Top-k: keeps the `k` highest-ranked logits and removes the rest. `k <= 0`
 * and `k >= V` switch the stage off. Exactly `k` positions are kept when
 * logits tie at the boundary; removed tokens stay removed.
 */
export function topKFilter(logits: readonly number[], k: number): number[] {
  if (k <= 0 || k >= logits.length) return [...logits];
  const out = [...logits];
  for (const token of ranking(logits).slice(k)) out[token] = Number.NEGATIVE_INFINITY;
  return out;
}

/**
 * Top-p: takes the softmax of the current logits, ranks the tokens by
 * decreasing probability, and keeps a token exactly when the mass ranked
 * strictly before it is below `p`. The token that carries the mass to or past
 * `p` is kept. `p = 1` (or more) switches the stage off: nothing is removed.
 */
export function topPFilter(logits: readonly number[], p: number): number[] {
  if (p >= 1) return [...logits];
  const probabilities = softmax(logits);
  const out = [...logits];
  let before = 0;
  for (const token of ranking(probabilities)) {
    if (!(before < p)) out[token] = Number.NEGATIVE_INFINITY;
    before += probabilities[token] ?? 0;
  }
  return out;
}

/** Temperature: divides every logit by `temperature`. Removed tokens stay removed. */
export function applyTemperature(logits: readonly number[], temperature: number): number[] {
  return logits.map((logit) => logit / temperature);
}

export interface SamplerSettings {
  /** `0`, or any value of at least the number of tokens, switches top-k off. */
  topK: number;
  /** `1` switches top-p off. */
  topP: number;
  temperature: number;
  order: StageOrder;
}

export interface SamplerResult {
  /** The distribution the chain samples from; 0 for a removed token. */
  probabilities: number[];
  /** For each token, the stage that removed it, or null when it was kept. */
  removedBy: (SamplerStage | null)[];
}

/**
 * Runs top-k, top-p, and temperature over one row in the given order: the
 * temperature after the two filters (`filters-first`) or before them. Top-p
 * always works on what top-k left.
 */
export function runSampler(logits: readonly number[], settings: SamplerSettings): SamplerResult {
  const { topK, topP, temperature, order } = settings;
  const start = order === "temperature-first" ? applyTemperature(logits, temperature) : [...logits];
  const afterTopK = topKFilter(start, topK);
  const afterTopP = topPFilter(afterTopK, topP);
  const end = order === "filters-first" ? applyTemperature(afterTopP, temperature) : afterTopP;
  const removed = (row: readonly number[], token: number) => row[token] === Number.NEGATIVE_INFINITY;
  return {
    probabilities: softmax(end),
    removedBy: logits.map((_, token) => {
      if (removed(start, token)) return null;
      if (removed(afterTopK, token)) return "top-k";
      return removed(afterTopP, token) ? "top-p" : null;
    }),
  };
}

/** How many tokens a result keeps. */
export function keptCount(result: SamplerResult): number {
  return result.removedBy.filter((stage) => stage === null).length;
}
