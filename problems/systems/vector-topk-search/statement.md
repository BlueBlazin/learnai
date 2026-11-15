Find the `k` nearest corpus vectors of every query by brute force: score each query
against every vector, and keep the best `k`. Exact search is one matrix product and a
selection. What needs care is the selection: it should not sort what it throws away,
and when two vectors score the same, the answer must not depend on the mood of the
selection routine.

## Specification

```python
def search(queries, corpus, k, metric="cosine"): ...
```

The **score** of query $q$ and corpus vector $c$ is

$$
s(q, c) = \begin{cases}
\dfrac{q \cdot c}{\lVert q \rVert_2 \, \lVert c \rVert_2} & \texttt{metric == "cosine"} \\[2ex]
q \cdot c & \texttt{metric == "dot"}
\end{cases}
$$

and a higher score means a closer vector.

- `queries`: float32, shape `(Q, D)`. `corpus`: float32, shape `(N, D)`. No vector
  is zero.
- `k`: a Python int, $k \ge 1$. Let $K = \min(k, N)$.
- `metric`: `"cosine"` (the default) or `"dot"`. No other value is passed.

The function returns the tuple `(indices, scores)`:

- `indices`: an integer tensor of shape `(Q, K)`. Row $i$ holds the indices of the
  $K$ corpus vectors with the highest scores for query $i$, **best first**.
- `scores`: a float32 tensor of shape `(Q, K)` with
  `scores[i, j]` $= s(\texttt{queries}[i], \texttt{corpus}[\texttt{indices}[i, j]])$.

**Ties.** Vectors with equal scores are ranked by index, lower first. This decides
both the order inside a row and which of several equal vectors make it into the top
$K$: the result is the first $K$ entries of the list of all $N$ indices sorted by
(score descending, index ascending).

The scores come from tensor operations over the whole corpus; a Python loop over
corpus vectors is too slow (see Constraints).

## Shapes

| name | shape | dtype |
|---|---|---|
| `queries` | `(Q, D)` | float32 |
| `corpus` | `(N, D)` | float32 |
| `indices` | `(Q, min(k, N))` | integer |
| `scores` | `(Q, min(k, N))` | float32 |

## Example

```python
corpus = [[1, 0], [0, 2], [3, 3], [2, 0], [-1, 0]]
queries = [[1, 0], [1, 1]]

search(queries, corpus, k=2)
# indices [[0, 3], [2, 0]]
# scores  [[1.0, 1.0], [1.0, 0.7071]]

search(queries, corpus, k=2, metric="dot")
# indices [[2, 3], [2, 1]]
# scores  [[3.0, 2.0], [6.0, 2.0]]
```

For the query `[1, 0]`, vectors 0 and 3 point in the same direction as the query:
both have cosine 1, and the lower index comes first. Under `"dot"` the long vectors
win instead: `[3, 3]` scores 3 and `[2, 0]` scores 2.

For the query `[1, 1]`, vector 2 has cosine 1. Vectors 0, 1, and 3 all have cosine
$1/\sqrt{2}$; only one place is left, and it goes to index 0. Under `"dot"` vector 2
scores 6, and vectors 1 and 3 both score 2: the second place goes to index 1.

## Constraints

- $1 \le Q \le 64$, $1 \le N \le 20{,}000$, $1 \le D \le 64$, $1 \le k \le 100$; `k`
  may exceed `N`.
- Scores are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. Indices are compared exactly. Where two vectors'
  scores differ by less than float32 can resolve (`1e-6`), either order is accepted;
  the tie cases use vectors whose scores are exact (small integers, halves), and
  there the tie rule is enforced.
- **Speed.** For 4 queries against 20,000 vectors of 64 dimensions with `k = 10`,
  the function must run within 10 times the reference's time in the same framework.
  The reference is one matrix product and a selection. Computing the scores one
  corpus vector at a time is about 90 times slower. Sorting every row in full is
  within the limit (2 to 4 times slower on a CPU), but selecting is the better
  habit: at a million vectors the sort is what you wait for.

## Why this matters

Every retrieval system starts here. Below a few hundred thousand vectors, exact
search over a matrix that fits in memory is the right index: it has perfect recall,
no build step, and nothing to tune, and one matrix product per batch of queries is
what the hardware does best. Above that size it is the ground truth that every
approximate index (IVF, HNSW, quantized codes) is measured against, and the
reranking step that those indexes end with. The tie rule is not pedantry: document
chunks are often duplicated, duplicates score identically, and a search whose
answer changes between runs or between a CPU and a GPU makes a retrieval bug
impossible to reproduce.
