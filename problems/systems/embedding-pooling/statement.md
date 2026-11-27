Implement two functions that turn token-level encoder outputs into fixed-size embeddings for vector search. The first pools token states into one unit-length vector per sequence using mean, first-token, or last-token pooling while ignoring padding. The second truncates embeddings to a shorter prefix and renormalizes them.

## Specification

```python
def pool(states, mask, mode): ...
def truncate_matryoshka(e, dim): ...
```

### Batch layout

`states` holds token representations for a batch of $B$ sequences of length $T$ with embedding dimension $D$. Specifically, `states[b, t]` is the vector at position `t` of sequence `b`. `mask[b, t]` is `1` when that position holds a real token and `0` when it is padding.

The real tokens in each sequence are contiguous and flush against one end, either the start (**right padding**) or the end (**left padding**). Every sequence has at least one real token. Values at padded positions are arbitrary and must not affect the output.

### `pool(states, mask, mode)`

For each sequence $b$, let $R_b = \{t : \texttt{mask}[b, t] = 1\}$ be the set of real token positions. Compute one pooled vector $p_b$ per sequence according to `mode`:

| `mode` | Pooled vector $p_b$ |
|---|---|
| `"mean"` | $\dfrac{1}{\lvert R_b \rvert} \sum_{t \in R_b} \texttt{states}[b, t]$ |
| `"cls"` | $\texttt{states}[b, \min R_b]$, the state of the **first** real token |
| `"last"` | $\texttt{states}[b, \max R_b]$, the state of the **last** real token |

No other value of `mode` is passed. After pooling, L2-normalize each vector with a floor of $10^{-12}$ on the norm:

$$
\operatorname{normalize}(p) = \frac{p}{\max(\lVert p \rVert_2,\ 10^{-12})} .
$$

With this floor, an all-zero vector stays zero instead of producing NaN. Every other vector is scaled to unit length.

### `truncate_matryoshka(e, dim)`

Matryoshka embeddings are trained so that prefixes of the full vector remain valid embeddings on their own. Given a tensor `e` whose last axis has size $D$ and whose leading axes are arbitrary batch axes (one or more axes total), keep the first `dim` coordinates along the last axis and renormalize them with the same rule:

$$
\operatorname{truncate}(e, \texttt{dim}) = \operatorname{normalize}(e_{1..\texttt{dim}}) .
$$

Here $1 \le \texttt{dim} \le D$.

## Shapes

| Name | Shape | Dtype |
|---|---|---|
| `states` | `(B, T, D)` | float32 |
| `mask` | `(B, T)` | integer, values 0 and 1 |
| `pool(...)` | `(B, D)` | float32 |
| `e` | `(..., D)` | float32 |
| `truncate_matryoshka(...)` | `(..., dim)` | float32 |

## Example

Consider a batch of two sequences with `T = 3` and `D = 2`. Sequence 0 has two real tokens and is right-padded with `[9, 0]`.

```python
states = [[[1, 0], [3, 4], [9, 0]], [[0, 2], [4, 0], [2, 1]]]
mask = [[1, 1, 0], [1, 1, 1]]

pool(states, mask, "mean")  # [[0.7071, 0.7071], [0.8944, 0.4472]]
pool(states, mask, "cls")  # [[1.0, 0.0],       [0.0, 1.0]]
pool(states, mask, "last")  # [[0.6, 0.8],       [0.8944, 0.4472]]
```

- In `"mean"` mode, sequence 0 averages only its two real tokens, giving $([1, 0] + [3, 4]) / 2 = [2, 2]$, which normalizes to $[0.7071, 0.7071]$. Including the padded token `[9, 0]` would produce $[13, 4] / 3$, pointing in a different direction.
- In `"last"` mode, the last real token of sequence 0 is at position 1 with state `[3, 4]` and length 5, so the normalized output is `[0.6, 0.8]`. Position 2 is ignored because it is padding.

Left-padding the same sequence produces the exact same embeddings:

```python
states = [[[9, 0], [1, 0], [3, 4]]]
mask = [[0, 1, 1]]
pool(states, mask, "cls")  # [[1.0, 0.0]] (first real token is at position 1)
pool(states, mask, "last")  # [[0.6, 0.8]]
```

Truncating a 4-dimensional unit vector to its first 2 coordinates requires renormalization:

```python
truncate_matryoshka([0.5, 0.5, 0.5, 0.5], 2)  # [0.7071, 0.7071]
```

The prefix `[0.5, 0.5]` has length $0.7071$ rather than 1, so dividing by its norm restores unit length.

## Constraints

- $1 \le B \le 64$, $1 \le T \le 64$, and $1 \le D \le 64$.
- All sequences in a batch are padded on the same side. A sequence may contain no padding at all.
- Outputs are compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`.
- **Padding invariance.** The same texts encoded by the kit's `TinyEncoder` (whose real-token states do not depend on padding) and padded on the left, on the right, or to a greater length must produce the same embeddings in every mode within an absolute tolerance of `1e-4`.
- No Python loop over sequences or token positions is needed.

## Why this matters

Pooling sits between the encoder and the vector index in every dense retrieval pipeline. The choice of pooling is tied to how the model was trained. BERT-style encoders typically use mean or `[CLS]` pooling, whereas decoder-based embedding models use last-token pooling. Using the wrong mode produces valid-looking vectors with poor retrieval quality.

Padding bugs are even harder to catch because they depend on batch composition. When padded tokens leak into the pooled vector, a document embedded in a batch with longer texts receives a different vector than the same text embedded alone. Offline batch indexing then silently disagrees with online single-query serving. Similarly, Matryoshka truncation lets a single model serve both full-size and compact indexes, but forgetting to renormalize leaves vectors with unequal lengths, which breaks cosine similarity under inner-product search.
