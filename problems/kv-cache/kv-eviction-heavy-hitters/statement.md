Implement one step of heavy-hitter eviction for a KV cache with a fixed memory budget. Track the total attention each cached token has received, and whenever the cache exceeds the budget, keep the newest tokens along with the highest-scoring older ones. Because each attention head evicts independently, the cache is no longer a contiguous prefix of the sequence, and scores must stay aligned with their entries.

![This problem is the KV cache, here kept within a budget by evicting low-scoring tokens.](assets/map.svg)

## Specification

```python
def h2o_step(cache, scores, attn_weights, budget, recent): ...  # -> (cache, scores, kept)
```

`h2o_step` runs after an attention step on one layer. Here `B` is the batch size, `H` is the number of heads, `S` is the number of entries currently in the cache (including the `T` tokens just appended), and `D` is the head dimension.

Inputs:

- `cache` is a tuple `(k, v)` of the keys and values the step attended over, in sequence order. Its last `T` entries belong to the `T` tokens appended in this step.
- `attn_weights` gives the attention distributions of the `T` new queries over the `S` cached entries. Row $i$ is the distribution of new query $i$ (non-negative, summing to `1`, and `0` for entries after its own position).
- `scores` holds the accumulated attention score of each of the `S - T` entries that were in the cache before this step. On the first call ($S = T$), `scores` is `None`.

Perform the following three steps:

**1. Accumulate.** Each entry's score is the total attention it has received from all queries since it was appended. Newly appended entries start at `0`. Add the column sums of `attn_weights` to the scores, independently for each batch item and head:

$$
\text{score}_j \leftarrow \text{score}_j + \sum_{i=0}^{T-1} \text{attn\_weights}_{i,j},
\qquad 0 \le j < S.
$$

**2. Evict.** If $S \le \text{budget}$, keep every entry. Otherwise, each head (in each batch item) keeps exactly `budget` entries:

- **Recent window:** the last `recent` positions, $S - \text{recent}, \dots, S - 1$, regardless of their scores.
- **Heavy hitters:** among the older positions $0, \dots, S - \text{recent} - 1$, the $\text{budget} - \text{recent}$ positions with the highest accumulated scores. If two positions have equal scores, the lower position index (the older token) ranks first.

**3. Return** `(cache, scores, kept)` with $N = \min(S, \text{budget})$:

- `kept`: an integer tensor of the positions (indices into axis `2` of the input cache) that each head keeps, in **increasing order**.
- `cache`: `(k, v)` gathered at `kept` so that entry $n$ of head $h$ in batch item $b$ is input entry `kept[b, h, n]`, preserving sequence order.
- `scores`: the accumulated scores of the kept entries in the same order, to be passed into the next call.

Positions index the current cache rather than original token numbers. After the first eviction, position $j$ in two different heads may hold different tokens.

Edge cases and rules:

- `budget` is a Python `int` $\ge 1$, and `recent` is a Python `int` with $0 \le \text{recent} \le \text{budget}$. Both stay constant across calls on the same cache.
- When `recent = budget`, the policy is a sliding window. When `recent = 0`, scores alone decide and a newly appended token can be evicted in the same step.
- `T` may be any integer $\ge 1$, including `T > budget` (trimming a long prompt down to the budget in one call). All `T` query rows contribute to the scores before eviction.
- Do not modify any input tensors in place. Return new tensors.

## Shapes

| name | shape | dtype |
|---|---|---|
| `k`, `v` in `cache` | `(B, H, S, D)` | float32 |
| `scores` | `(B, H, S - T)`, or `None` when $S = T$ | float32 |
| `attn_weights` | `(B, H, T, S)` | float32 |
| `kept` | `(B, H, N)`, $N = \min(S, \text{budget})$ | integer |
| `k`, `v` returned | `(B, H, N, D)` | float32 |
| `scores` returned | `(B, H, N)` | float32 |

## Example

Consider `B = 1`, `H = 1`, `budget = 4`, and `recent = 2`. Tokens are numbered from `0`.

**Prefill.** Four tokens arrive at once, so `scores` is `None` and `attn_weights` is the causal attention matrix of the prompt:

```python
attn_weights = [
    [1.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.25, 0.25, 0.5, 0.0],
    [0.5, 0.125, 0.125, 0.25],
]
```

The column sums give the scores `[2.25, 0.875, 0.625, 0.25]`. All four entries fit within the budget, so `kept = [0, 1, 2, 3]`.

**Step 1.** Token 4 is appended and attends with `[0.125, 0.125, 0.5, 0.125, 0.125]`.

| position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| token | 0 | 1 | 2 | 3 | 4 |
| score before | 2.25 | 0.875 | 0.625 | 0.25 | 0 (new) |
| score after | 2.375 | 1.0 | 1.125 | 0.375 | 0.125 |

Five entries exceed the budget of `4`. Positions `3` and `4` form the recent window. Among positions `0`, `1`, and `2`, the two highest scores are `2.375` (position `0`) and `1.125` (position `2`). Position `1` is evicted even though its score (`1.0`) is higher than both window entries. The output is `kept = [0, 2, 3, 4]` with scores `[2.375, 1.125, 0.375, 0.125]`.

**Step 2.** Token 5 is appended and attends with `[0.25, 0.0, 0.5, 0.125, 0.125]` over the five entries now in the cache.

| position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| token | 0 | 2 | 3 | 4 | 5 |
| score before | 2.375 | 1.125 | 0.375 | 0.125 | 0 (new) |
| score after | 2.625 | 1.125 | 0.875 | 0.25 | 0.125 |

Positions `3` and `4` (tokens 4 and 5) form the recent window. Among positions `0`, `1`, and `2`, position `2` (token 3) has the lowest accumulated score and is evicted. The output is `kept = [0, 1, 3, 4]` (holding tokens 0, 2, 4, and 5) with scores `[2.625, 1.125, 0.25, 0.125]`.

Notice that token 2 received `0.0` attention in this step yet survives because of its accumulated score, while token 3 received the highest weight in this step (`0.5`) yet is evicted.

## Constraints

- $1 \le B \le 2$, $1 \le H \le 4$, $D \le 8$, $S \le 16$, and $1 \le \text{budget} \le 8$.
- `kept` is compared exactly. Returned keys and values are copies of the input entries and are compared exactly. Returned scores are compared with relative tolerance `1e-4` and absolute tolerance `1e-5`.
- Whenever two competing scores are not equal, they differ by at least $10^{-3}$, so floating-point rounding cannot change the ranking. Exact ties do occur (constructed from sums of powers of two) and must be broken in favor of the lower position index.
- No Python loops over batch rows, heads, or positions are needed.

## Why this matters

A KV cache grows by one entry per layer for every generated token, and long generations eventually run out of memory. Keeping only a sliding window of recent tokens is simple, but it discards older context the model still references. Heavy-Hitter Oracle (H2O) exploits the fact that attention is heavily skewed. A small subset of tokens accumulates most of the attention weight over time, and keeping those heavy hitters alongside the most recent tokens preserves generation quality at a fraction of the cache size.

Getting the alignment right is critical because eviction bugs fail silently. If scores are not gathered with the kept entries, they describe the wrong tokens after the very first eviction. Ranking by only the latest step's attention evicts tokens the model consults periodically. Omitting the recent window evicts newly generated tokens immediately before later queries can attend to them. None of these bugs crash the run, yet all of them quietly degrade model output.
