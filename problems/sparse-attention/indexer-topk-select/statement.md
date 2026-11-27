Implement the two halves of a sparse-attention indexer. The first function scores every query-key pair without normalizing the scores. The second function selects up to `k` visible keys with the highest scores for each query, using deterministic tie-breaking.

![This problem is the indexer's scores and its top-k, which choose the tokens each query attends to.](assets/map.svg)

## Specification

```python
def index_scores(q_idx, k_idx, w): ...  # -> float32 (T, S)
def topk_mask(scores, visible, k): ...  # -> bool (T, S)
```

**`index_scores`.** Suppose there are $T$ query tokens, $S$ key tokens, and $H$ indexer heads of width $D$. Query token $t$ has one query vector per head, $q_{t,h} \in \mathbb{R}^D$, and one scalar weight per head, $w_{t,h} \in \mathbb{R}$. Key token $s$ has a single key vector $k_s \in \mathbb{R}^D$ shared across all heads.

Each head computes a scaled dot product between its query and the shared key, applies ReLU, and multiplies by $w_{t,h} / \sqrt{H}$. Summing over all heads gives the score $I_{t,s}$ of key $s$ for query $t$:

$$
I_{t,s} \;=\; \sum_{h=0}^{H-1} \frac{w_{t,h}}{\sqrt{H}}\;
\operatorname{relu}\!\left(\frac{q_{t,h} \cdot k_s}{\sqrt{D}}\right).
$$

The ReLU runs on each head's scaled dot product before the weighted sum. Head weights are used as given and may be negative, so a head with a negative weight lowers the score. `index_scores` returns $I$ for every query-key pair and applies no mask.

**`topk_mask`.** Given a score matrix `scores`, a boolean mask `visible`, and a Python integer $k \ge 1$ (which may exceed $S$), `topk_mask` returns a boolean mask selecting the highest-scoring visible keys for each query. In a causal attention layer, `visible[t, s]` indicates that key `s` is not after query `t`, though `visible` may have any boolean pattern.

Let $n_t$ be the number of `True` entries in row $t$ of `visible`. Row $t$ of the output is `True` at the $\min(k, n_t)$ visible entries with the highest scores, and `False` everywhere else:

- An invisible entry is never selected and never takes a slot from a visible entry, regardless of its score.
- Ties break toward the lower key index. `0.0` and `-0.0` are equal. Scores that differ are never treated as equal, no matter how close they are, and equal scores remain equal at any magnitude.
- If $n_t \le k$, row $t$ selects every visible entry. If $n_t = 0$, row $t$ is all `False`.
- Do not modify any input tensor in place.

## Shapes

| name | shape | dtype |
|---|---|---|
| `q_idx` | `(T, H, D)` | float32 |
| `k_idx` | `(S, D)` | float32 |
| `w` | `(T, H)` | float32 |
| result of `index_scores`, `scores` | `(T, S)` | float32 |
| `visible`, result of `topk_mask` | `(T, S)` | bool |

## Example

Consider two queries, three keys, and $H = 2$ heads of width $D = 4$.

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

For query 0, head 0 has raw dot products $(2, -2, 4)$, which scale by $1/\sqrt{4}$ to $(1, -1, 2)$ and become $(1, 0, 2)$ after ReLU. Head 1 produces $(1, 3, 0)$. Both head weights are $1/\sqrt{2}$, so the final scores are $(2, 3, 2)/\sqrt{2}$. Without the ReLU, key 1 would score $2/\sqrt{2}$.

For query 1, head 0 produces $(2, 2, 2)$ with weight $2/\sqrt{2}$, and head 1 produces $(0, 0, 1)$ with weight $-1/\sqrt{2}$. Summing them gives $(4, 4, 3)/\sqrt{2}$, so the negative weight costs key 2 its tie with keys 0 and 1.

With $k = 1$, query 0 selects key 1, its highest-scoring visible key. Query 1 has two tied best scores and picks the lower index, key 0. With $k = 3$, query 0 has only two visible keys (key 2 is invisible) and selects both.

## Constraints

- $1 \le T, S \le 64$, $1 \le H \le 32$, $1 \le D \le 128$, and $1 \le k \le 4096$.
- Every input value is finite, including invisible entries in `scores`, with $|\texttt{scores}| \le 10^4$.
- `index_scores` is checked against the reference implementation and, on one input, against `GlmMoeDsaIndexer` from Hugging Face Transformers (relative tolerance `1e-4`, absolute tolerance `1e-5`).
- `topk_mask` is checked for exact equality and must return a boolean tensor. Because `scores` is passed directly as an input, the expected mask does not depend on floating-point rounding.
- When `k` is at least the largest $n_t$, the result of `topk_mask` equals `visible`.

## Why this matters

DeepSeek Sparse Attention (used in DeepSeek-V3.2 and GLM-5.3) runs this indexer in front of every sparse attention layer. GLM-5.3 scores keys with 32 heads of width 128 and keeps 2048 tokens per query regardless of context length.

Because the attention layer attends only to the selected tokens, a selection that is off by even one token silently changes the output without raising an error. Running top-k before masking future tokens, normalizing or applying ReLU to the wrong quantity, or breaking ties differently across inference engines all produce a model that runs cleanly while diverging from the trained weights.
