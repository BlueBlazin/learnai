An embedding model returns one state per token; a search index wants one vector per
text. Implement the step in between: pool the token states of each row into a single
vector (mean, first token, or last token), normalize it to unit length, and shorten a
finished embedding to its first dimensions. Each of the three poolings is one line.
All of them go wrong in the same place, the padding, and each in its own way.

## Specification

```python
def pool(states, mask, mode): ...
def truncate_matryoshka(e, dim): ...
```

**The batch.** `states[b, t]` is the encoder's state for position `t` of row `b`.
`mask[b, t]` is `1` if that position holds a real token and `0` if it is padding.
The real tokens of a row are contiguous and touch one end of the row: its start
(**right** padding) or its end (**left** padding). Every row has at least one real
token. The states at padded positions are arbitrary numbers and must not influence
the result.

**`pool(states, mask, mode)`** computes one vector $p_b$ per row, then normalizes it.
With $R_b = \{t : \texttt{mask}[b, t] = 1\}$, the real positions of row $b$:

| `mode` | pooled vector $p_b$ |
|---|---|
| `"mean"` | $\dfrac{1}{\lvert R_b \rvert} \sum_{t \in R_b} \texttt{states}[b, t]$ |
| `"cls"` | $\texttt{states}[b, \min R_b]$: the state of the **first** real token |
| `"last"` | $\texttt{states}[b, \max R_b]$: the state of the **last** real token |

No other mode is passed. The result is the L2-normalized vector, with a floor on the
divisor:

$$
\operatorname{normalize}(p) = \frac{p}{\max(\lVert p \rVert_2,\ 10^{-12})} .
$$

A vector that is exactly zero therefore stays zero; every other vector comes out
with length 1.

**`truncate_matryoshka(e, dim)`** shortens embeddings that were trained so that
their prefixes are embeddings too. It keeps the first `dim` coordinates along the
last axis and normalizes again with the same rule:

$$
\operatorname{truncate}(e, d) = \operatorname{normalize}(e_{1..d}) .
$$

`e` has one or more axes; all but the last are batch axes. $1 \le$ `dim` $\le D$.

## Shapes

| name | shape | dtype |
|---|---|---|
| `states` | `(B, T, D)` | float32 |
| `mask` | `(B, T)` | integer, values 0 and 1 |
| `pool(...)` | `(B, D)` | float32 |
| `e` | `(..., D)` | float32 |
| `truncate_matryoshka(...)` | `(..., dim)` | float32 |

## Example

Two rows, `T = 3`, `D = 2`. Row 0 has two real tokens and is padded on the right
with a state of `[9, 0]`:

```python
states = [[[1, 0], [3, 4], [9, 0]], [[0, 2], [4, 0], [2, 1]]]
mask = [[1, 1, 0], [1, 1, 1]]

pool(states, mask, "mean")  # [[0.7071, 0.7071], [0.8944, 0.4472]]
pool(states, mask, "cls")  # [[1.0, 0.0],       [0.0, 1.0]]
pool(states, mask, "last")  # [[0.6, 0.8],       [0.8944, 0.4472]]
```

- `"mean"`, row 0: $([1, 0] + [3, 4]) / 2 = [2, 2]$, normalized to
  $[0.7071, 0.7071]$. Averaging all three positions would give $[13, 4] / 3$,
  a different direction.
- `"last"`, row 0: the last real token is at position 1, state $[3, 4]$, length 5.
  Position 2 is padding.

The same text padded on the left gives the same embeddings:

```python
states = [[[9, 0], [1, 0], [3, 4]]]
mask = [[0, 1, 1]]
pool(states, mask, "cls")  # [[1.0, 0.0]]: the first real token is at position 1
pool(states, mask, "last")  # [[0.6, 0.8]]
```

```python
truncate_matryoshka([0.5, 0.5, 0.5, 0.5], 2)  # [0.7071, 0.7071]
```

The prefix `[0.5, 0.5]` has length $0.7071$, not 1; dividing by it restores unit
length.

## Constraints

- $1 \le B \le 64$, $1 \le T \le 64$, $1 \le D \le 64$. All rows of a batch are
  padded on the same side. A row may have no padding at all.
- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`.
- **Padding never changes the embedding.** The same texts, encoded by the kit's
  `TinyEncoder` (whose states for real tokens do not depend on the padding) and
  padded on the left, on the right, or to a greater length, must give the same
  embeddings in every mode, to an absolute tolerance of `1e-4`.
- No Python loop over rows or positions is needed.

## Why this matters

This function sits between the embedding model and the vector index in every
retrieval system, and it is model-specific: BERT-style encoders are trained for
mean or `[CLS]` pooling, decoder-based embedding models for last-token pooling,
and using the wrong one gives vectors that look fine and retrieve badly. The
padding bugs are worse because they depend on the batch: a text embedded alone and
the same text embedded next to a longer one get different vectors, so the index
built by a batch job does not match the queries embedded one at a time. Matryoshka
truncation is how one model serves a 1024-dimensional index and a 256-dimensional
one; forgetting to renormalize turns every cosine similarity into an inner product
of vectors of different lengths.
