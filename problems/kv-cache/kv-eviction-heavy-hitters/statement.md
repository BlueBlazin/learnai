Implement one step of heavy-hitter eviction for a KV cache with a fixed budget: keep a
running total of the attention every cached token has received, and when the cache
exceeds the budget, keep the newest tokens and the highest-scoring older ones. Each
head evicts on its own, so the cache stops being a prefix of the sequence, and the
scores have to move with the entries they belong to.

![This problem is the KV cache, here kept within a budget by evicting low-scoring tokens.](assets/map.svg)

## Specification

```python
def h2o_step(cache, scores, attn_weights, budget, recent): ...  # -> (cache, scores, kept)
```

The function runs after an attention step, on one layer. At that point:

- `cache` is a tuple `(k, v)` of float32 tensors of shape `(B, H, S, D)`: the keys and
  values the step attended over, in sequence order. Its last `T` entries belong to
  the `T` tokens this step appended.
- `attn_weights` is a float32 tensor of shape `(B, H, T, S)`: row $i$ is the attention
  distribution of new query $i$ over the $S$ entries (non-negative, summing to 1,
  and 0 for entries after its own).
- `scores` is a float32 tensor of shape `(B, H, S - T)`: the accumulated score of each
  entry that was in the cache before this step. It is `None` on the first call, when
  there is no such entry ($S = T$).

**1. Accumulate.** The score of an entry is the total attention it has received from
every query since it was appended. A new entry starts at 0, and this step adds its
column sums:

$$
\text{score}_j \leftarrow \text{score}_j + \sum_{i=0}^{T-1} \text{attn\_weights}_{i,j},
\qquad 0 \le j < S,
$$

separately for every batch row and head.

**2. Evict.** If $S \le \text{budget}$, every entry is kept. Otherwise each head (of
each batch row) keeps exactly `budget` entries:

- the **recent window**: the last `recent` positions, $S - \text{recent}, \dots, S - 1$,
  whatever their scores;
- the **heavy hitters**: of the other positions $0, \dots, S - \text{recent} - 1$, the
  $\text{budget} - \text{recent}$ with the highest accumulated scores. Of two
  positions with equal scores, the lower one (the older token) ranks first.

**3. Return** a tuple `(cache, scores, kept)` with $N = \min(S, \text{budget})$:

- `kept`: an integer tensor of shape `(B, H, N)`, the positions (indices into axis 2
  of the cache that was passed in) that each head keeps, in **increasing order**.
- `cache`: `(k, v)` with shape `(B, H, N, D)`: entry $n$ of head $h$ in row $b$ is
  entry `kept[b, h, n]` of the input. Sequence order is preserved.
- `scores`: float32 of shape `(B, H, N)`, the accumulated scores of the kept entries,
  in the same order. The next call receives these.

Positions are positions in the current cache, not token numbers: after the first
eviction, position $j$ of two heads may hold different tokens.

Details:

- `budget` is a Python int $\ge 1$ and `recent` a Python int with
  $0 \le \text{recent} \le \text{budget}$; both are the same in every call on one
  cache. With `recent = budget` the policy is a sliding window; with `recent = 0`
  scores alone decide, and a new token can be evicted in the step that appended it.
- `T` may be any size $\ge 1$, also larger than `budget` (a long prompt is cut down
  to the budget in one call). All `T` rows count toward the scores before anything
  is evicted.
- The function returns new tensors and does not modify the ones it was given.

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

One batch row, one head, `budget = 4`, `recent = 2`. Tokens are numbered from 0.

**Prefill.** Four tokens arrive at once, so `scores` is `None` and `attn_weights` is
the causal attention matrix of the prompt:

```python
attn_weights = [
    [1.0, 0.0, 0.0, 0.0],
    [0.5, 0.5, 0.0, 0.0],
    [0.25, 0.25, 0.5, 0.0],
    [0.5, 0.125, 0.125, 0.25],
]
```

The column sums are the scores, `[2.25, 0.875, 0.625, 0.25]`. Four entries fit the
budget: `kept = [0, 1, 2, 3]`.

**Step 1.** Token 4 is appended and attends with `[0.125, 0.125, 0.5, 0.125, 0.125]`.

| position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| token | 0 | 1 | 2 | 3 | 4 |
| score before | 2.25 | 0.875 | 0.625 | 0.25 | 0 (new) |
| score after | 2.375 | 1.0 | 1.125 | 0.375 | 0.125 |

Five entries exceed the budget. Positions 3 and 4 are the recent window. Of positions
0, 1, 2, the two highest scores are kept: 2.375 and 1.125. Position 1 is evicted,
although its score (1.0) is higher than those of both window entries.
`kept = [0, 2, 3, 4]`, and the returned scores are `[2.375, 1.125, 0.375, 0.125]`.

**Step 2.** Token 5 is appended and attends with `[0.25, 0.0, 0.5, 0.125, 0.125]`
over the five entries the cache now holds.

| position | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| token | 0 | 2 | 3 | 4 | 5 |
| score before | 2.375 | 1.125 | 0.375 | 0.125 | 0 (new) |
| score after | 2.625 | 1.125 | 0.875 | 0.25 | 0.125 |

Positions 3 and 4 (tokens 4 and 5) are the window. Of positions 0, 1, 2, position 2
(token 3) has the lowest accumulated score and is evicted: `kept = [0, 1, 3, 4]`, the
cache holds tokens 0, 2, 4, 5, and the scores are `[2.625, 1.125, 0.25, 0.125]`.

Token 2 received nothing in this step and still survives: its accumulated score
decides, not its latest one. Token 3 had just received the most attention of any
entry and is evicted.

## Constraints

- $1 \le B \le 2$, $1 \le H \le 4$, $D \le 8$, $S \le 16$, $1 \le \text{budget} \le 8$.
- `kept` is compared exactly. Keys and values are copies of the input and are
  compared exactly. Scores are compared to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`.
- Where two competing scores are not exactly equal, they differ by at least
  $10^{-3}$, so float rounding cannot change a ranking. Exact ties occur (with
  weights that are sums of powers of two) and must be resolved as specified.
- No Python loops over batch rows, heads, or positions are needed.

## Why this matters

A KV cache grows by one entry per layer and token for as long as a request runs, and
a long generation ends when it no longer fits. Truncating to the most recent tokens
is cheap and loses whatever the model was still looking at further back. H2O
observed that attention is concentrated: a small set of tokens, the heavy hitters,
receives most of the accumulated attention, and a cache made of those plus the recent
tokens preserves quality at a fraction of the size. The bookkeeping decides whether
that works. Scores that are not gathered along with the entries describe the wrong
tokens from the first eviction on. Ranking by the latest step's attention evicts the
token the model consults every tenth step. Leaving out the recent window evicts the
newest token immediately, since nothing has had a chance to attend to it yet. None of
these crash; the cache keeps its size and the output drifts.
