Implement the two halves of the indexer that decides which earlier tokens a sparse
attention layer attends to: a score for every query and key, and the selection of the
`k` best visible keys per query. The scores are never normalized, and the selection
has to say what happens to equal scores and to entries a query may not see.

## Specification

```python
def index_scores(q_idx, k_idx, w): ...  # -> float32 (T, S)
def topk_mask(scores, visible, k): ...  # -> bool (T, S)
```

**`index_scores`.** The indexer has $H$ heads of width $D$. Query token $t$ has one
indexer query per head, $q_{t,h} \in \mathbb{R}^D$, and one weight per head,
$w_{t,h} \in \mathbb{R}$. Key token $s$ has a single indexer key $k_s \in \mathbb{R}^D$
that all heads share. The score of key $s$ for query $t$ is

$$
I_{t,s} \;=\; \sum_{h=0}^{H-1} \frac{w_{t,h}}{\sqrt{H}}\;
\operatorname{relu}\!\left(\frac{q_{t,h} \cdot k_s}{\sqrt{D}}\right).
$$

The ReLU acts on each head's dot product, before the sum. The weights are used as
given: they may be negative, and a head with a negative weight lowers the score.
`index_scores` returns $I$ for every pair; it applies no mask.

**`topk_mask`.** Let $n_t$ be the number of True entries in row $t$ of `visible`. The
result is True at the $\min(k, n_t)$ visible entries of row $t$ with the highest
scores, and False everywhere else:

- An invisible entry is never selected and never takes the place of a visible one,
  whatever its score.
- Among equal scores the lower index is selected first. `0.0` and `-0.0` are equal.
  Scores that differ are not equal, however close they are, and equal scores are
  equal at any magnitude.
- A row with $n_t \le k$ selects exactly its visible entries; a row with $n_t = 0$
  selects nothing.

Arguments:

- `q_idx`: float32, shape `(T, H, D)`. `k_idx`: float32, shape `(S, D)`. `w`: float32,
  shape `(T, H)`.
- `scores`: float32, shape `(T, S)`, every entry finite (also the invisible ones).
- `visible`: bool, shape `(T, S)`, any pattern. In a causal layer, `visible[t, s]` says
  that key `s` is not after query `t`.
- `k`: a Python int, $k \ge 1$. It may exceed $S$.
- The tensors passed in are not modified.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q_idx` | `(T, H, D)` | float32 |
| `k_idx` | `(S, D)` | float32 |
| `w` | `(T, H)` | float32 |
| result of `index_scores`, `scores` | `(T, S)` | float32 |
| `visible`, result of `topk_mask` | `(T, S)` | bool |

## Example

Two queries, three keys, $H = 2$ heads of width $D = 4$:

```python
q_idx = [
    [[2, 0, 0, 0], [0, 2, 0, 0]],  # query 0: head 0, head 1
    [[2, 2, 0, 0], [0, 0, 2, 0]],  # query 1
]
k_idx = [[1, 1, 0, 0], [-1, 3, 0, 0], [2, 0, 1, 0]]
w = [[1, 1], [2, -1]]

scores = index_scores(q_idx, k_idx, w)
# [[1.414214, 2.121320, 1.414214],
#  [2.828427, 2.828427, 2.121320]]

visible = [[True, True, False], [True, True, True]]
topk_mask(scores, visible, 1)  # [[False, True, False], [True, False, False]]
topk_mask(scores, visible, 3)  # [[True, True, False], [True, True, True]]
```

Query 0: the dot products of head 0 are $(2, -2, 4)$, scaled by $1/\sqrt{4}$ to
$(1, -1, 2)$, and the ReLU gives $(1, 0, 2)$; head 1 gives $(1, 3, 0)$. Both weights
are $1/\sqrt{2}$, so the scores are $(2, 3, 2)/\sqrt{2}$. Without the ReLU, key 1 would
score $2/\sqrt{2}$.

Query 1: head 0 gives $(2, 2, 2)$ with weight $2/\sqrt{2}$, and head 1 gives
$(0, 0, 1)$ with weight $-1/\sqrt{2}$: the scores are $(4, 4, 3)/\sqrt{2}$. The negative
weight costs key 2 its tie with the others.

With $k = 1$, query 0 selects key 1. Key 2 is invisible to it. Query 1 has two equal
best scores and selects the lower index, key 0. With $k = 3$, query 0 has only two
visible keys and selects both.

## Constraints

- $1 \le T, S \le 64$, $1 \le H \le 32$, $1 \le D \le 128$, $1 \le k \le 4096$.
- Every input value is finite, with $|\texttt{scores}| \le 10^4$.
- `index_scores` is compared with the reference, and for one input with the scores
  computed by `GlmMoeDsaIndexer` of Hugging Face Transformers: relative tolerance
  `1e-4`, absolute tolerance `1e-5`.
- `topk_mask` is compared exactly. Its scores are an input, so its result does not
  depend on rounding. It must return a boolean tensor.
- With `k` at least the largest $n_t$, the result equals `visible`.

## Why this matters

DeepSeek Sparse Attention (DeepSeek-V3.2, GLM-5.3) runs this indexer in front of every
attention layer that has one: GLM-5.3 scores with 32 heads of width 128 and keeps 2048
tokens per query, however long the context is. The layer then attends over the
selected tokens only, so a selection that is wrong by one token changes the output
with no error anywhere. A top-k taken before the future is masked, an indexer that
normalizes or rectifies the wrong quantity, and a tie resolved differently in two
engines all produce a model that runs and answers slightly differently from the one
that was trained.
