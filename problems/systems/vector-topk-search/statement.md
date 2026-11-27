Implement exact brute-force vector search to find the `k` closest corpus vectors for each query under cosine similarity or inner product. Compute all query-corpus scores using vectorized tensor operations and return the top `k` results in descending score order with deterministic tie-breaking.

## Specification

```python
def search(queries, corpus, k, metric="cosine"): ...
```

`queries` contains $Q$ query vectors of dimension $D$, and `corpus` contains $N$ corpus vectors of dimension $D$. Every vector is nonzero. `k` is a Python integer with $k \ge 1$, and $K = \min(k, N)$ is the number of results returned per query. `metric` is either `"cosine"` (default) or `"dot"`. No other metric is passed.

The similarity score between query vector $q$ and corpus vector $c$ is

$$
s(q, c) = \begin{cases}
\dfrac{q \cdot c}{\lVert q \rVert_2 \, \lVert c \rVert_2} & \texttt{metric == "cosine"} \\[2ex]
q \cdot c & \texttt{metric == "dot"}
\end{cases}
$$

Higher scores indicate closer matches.

The function returns a tuple `(indices, scores)`:

- Row `i` of `indices` contains the corpus indices of the $K$ highest-scoring vectors for query `i`, ordered from best to worst.
- `scores[i, j]` equals $s(\texttt{queries}[i], \texttt{corpus}[\texttt{indices}[i, j]])$.

### Tie-breaking

When two corpus vectors have the exact same score for a query, the vector with the lower corpus index ranks first. This rule governs both the order within a row and which tied vectors make the top-$K$ cutoff. Equivalently, row `i` is the first $K$ entries of all $N$ corpus indices sorted by `(score descending, index ascending)`.

Compute scores using vectorized tensor operations over the full corpus. A Python loop over corpus vectors is too slow (see Constraints).

## Shapes

| Name | Shape | Dtype |
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

- For query `[1, 0]`, corpus vectors 0 (`[1, 0]`) and 3 (`[2, 0]`) both point in the same direction as the query and have cosine similarity 1.0. Because they tie, index 0 comes before index 3. Under `"dot"`, vector length matters, so `[3, 3]` scores 3.0 and `[2, 0]` scores 2.0.
- For query `[1, 1]`, corpus vector 2 (`[3, 3]`) has cosine similarity 1.0. Vectors 0, 1, and 3 all tie at cosine similarity $1/\sqrt{2} \approx 0.7071$, and the remaining slot goes to the lowest index, 0. Under `"dot"`, vector 2 scores 6.0, while vectors 1 and 3 tie at 2.0, so the second slot goes to index 1.

## Constraints

- $1 \le Q \le 64$, $1 \le N \le 20{,}000$, $1 \le D \le 64$, and $1 \le k \le 100$. `k` may exceed `N`.
- `scores` is compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. `indices` is compared exactly. When two vectors have scores within `1e-6` (below float32 resolution), either ordering is accepted. Tie-breaking test cases use exact scores (small integers or halves), where the tie rule is strictly enforced.
- **Speed.** On 4 queries against 20,000 vectors of dimension 64 with `k = 10`, your implementation must run within 10x of the reference time in the same framework. The reference uses one matrix multiplication and a top-k selection. Scoring one corpus vector at a time in Python is roughly 90x slower. A full sort of every row stays within the limit (2x to 4x slower on CPU), though top-k selection scales much better as `N` grows.

## Why this matters

Exact brute-force search is the foundation of every vector retrieval system. Up to a few hundred thousand vectors, a matrix multiplication over an in-memory corpus gives exact recall with no index build time and no tuning parameters. At larger scales, exact search serves both as the ground truth for evaluating approximate indexes (such as IVF, HNSW, or quantized indexes) and as the final reranking stage over candidate results.

Deterministic tie-breaking matters in production because real corpora frequently contain duplicate or near-duplicate chunks that receive identical scores. If tied rankings vary across runs, hardware backends, or library versions, retrieval regressions become impossible to reproduce.
