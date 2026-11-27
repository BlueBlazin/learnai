## The idea

Let `queries` have shape `(Q, D)` and `corpus` have shape `(N, D)`.

**Scoring.** All $Q \times N$ inner products between queries and corpus vectors are computed in a single matrix multiplication, `queries @ corpus.T`, producing a score table of shape `(Q, N)`. Cosine similarity is the inner product of unit-normalized vectors:

$$
\frac{q \cdot c}{\lVert q \rVert \lVert c \rVert}
= \frac{q}{\lVert q \rVert} \cdot \frac{c}{\lVert c \rVert} .
$$

Normalizing the rows of both matrices before multiplying therefore reuses the same matrix multiplication for cosine similarity. Matrix multiplication is heavily optimized on every backend (blocked, vectorized, multithreaded, or GPU-accelerated), which is why a Python loop computing one dot product at a time is orders of magnitude slower for the same arithmetic.

**Selecting.** Sorting an entire row costs $O(N \log N)$ and orders the $N - k$ entries that are immediately discarded. Top-k selection finds the $k$ largest values in $O(N)$ time (via quickselect) or $O(N \log k)$ time (via a size-$k$ heap), leaving only $k$ winners to sort.

**Handling ties.** Most selection primitives are not stable. For example, `torch.topk` does not guarantee index order among tied elements, and `mx.argpartition` leaves order within each partition unspecified. However, the *values* of the top $k$ scores are always well-defined.

To break ties deterministically in PyTorch, read only the $k$-th largest score $t$ for each row and build a boolean selection mask from it:

- Every entry with score $> t$ is included (there are strictly fewer than $k$ such entries).
- The remaining slots (`room = k - count(score > t)`) go to entries with score $= t$ in ascending index order. Taking a cumulative sum (`cumsum`) of the `score == t` mask along each row numbers tied entries 1, 2, 3, ... from left to right, so keeping those with count $\le \texttt{room}$ selects the lowest-index ties.

This mask has exactly $k$ true entries per row. Extracting their column indices produces the winning set in ascending index order, and a stable descending sort over those $k$ scores finishes the ranking.

## Reference solution (PyTorch)

```python
def search(queries, corpus, k, metric="cosine"):
    if metric == "cosine":
        queries = queries / queries.norm(dim=1, keepdim=True)
        corpus = corpus / corpus.norm(dim=1, keepdim=True)
    scores = queries @ corpus.T
    k = min(k, corpus.shape[0])

    kth = torch.topk(scores, k, dim=1).values[:, -1:]
    above = scores > kth
    tied = scores == kth
    room = k - above.sum(dim=1, keepdim=True)
    keep = above | (tied & (tied.cumsum(dim=1) <= room))
    index = keep.nonzero()[:, 1].view(-1, k)

    picked = scores.gather(1, index)
    order = torch.argsort(picked, dim=1, descending=True, stable=True)
    return index.gather(1, order), picked.gather(1, order)
```

`keep.nonzero()` returns coordinates in row-major order, so taking its column indices and reshaping to `(Q, k)` yields each row's selected indices in ascending order.

The JAX and MLX references differ slightly because of framework primitives:

- **JAX:** `jax.lax.top_k(scores, k)` returns both values and indices in descending order and is stable by default (`is_stable=True`), preserving lower indices first on ties. No extra tie-breaking step is needed.
- **MLX:** `mx.topk` returns values only, and `mx.argpartition` is unstable on ties. Because `mx.argsort` is stable and runs on the GPU (where sorting 20,000 entries per row costs roughly the same as partitioning), the reference uses `mx.argsort(-scores, axis=1)[:, :k]`. Negating `scores` rather than reversing an ascending sort is essential, because reversing an ascending sort would place higher indices first among ties.

## Common mistakes

- **Using raw dot products for `"cosine"`.** Without normalization, long vectors dominate every query. This bug hides when testing on already-normalized embeddings and surfaces later when a model returns unnormalized vectors.
- **Normalizing only one side.** Normalizing only `corpus` preserves the ranking (each query row is scaled by a constant positive factor), but returns incorrect similarity scores. Any downstream similarity threshold calibrated to true cosine scores will cut off in the wrong place.
- **Ignoring `metric`.** Models trained for inner-product search encode useful signal in vector magnitude, so normalizing when `metric == "dot"` changes the ranking.
- **Trusting the indices from an unstable `topk`.** The returned scores are correct, but the indices chosen among ties are arbitrary and can change across runs, devices, or library versions.
- **Using an unstable full sort.** `torch.sort` and `torch.argsort` are unstable by default. Pass `stable=True` to guarantee lower-index-first tie-breaking.
- **Forgetting to cap `k` at `N`.** `topk` raises an error when `k > N`, which happens in production on small or heavily filtered corpora.
- **Looping over the corpus in Python.** Performing the same dot products inside a Python loop issues $N$ separate framework calls and is roughly two orders of magnitude slower.

## Complexity

The matrix multiplication takes $O(QND)$ time, selection takes $O(QN)$ time, and sorting the top $k$ winners takes $O(Qk \log k)$ time. The full score table uses $O(QN)$ memory. For very large corpora, score the corpus in blocks and merge each block's top $k$ with the running best so memory stays bounded at $Q$ times the block size.

## In the wild

- Faiss implements exact brute-force search as `IndexFlatIP` (inner product) and `IndexFlatL2` (Euclidean distance). It serves as the recall baseline for evaluating approximate indexes and as the coarse quantizer over cluster centroids.
- Sentence Transformers' `semantic_search` computes exact search in chunks of queries and corpus vectors, calling `torch.topk` on each chunk.
- Vector databases often ask whether embeddings are pre-normalized because normalizing once at indexing time reduces cosine search to inner-product search.
