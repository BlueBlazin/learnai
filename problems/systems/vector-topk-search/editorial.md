## The idea

**Scoring.** The inner products of all queries with all corpus vectors are the
entries of one matrix product, `queries @ corpus.T`, of shape `(Q, N)`. Cosine
similarity is the inner product of the normalized vectors,

$$
\frac{q \cdot c}{\lVert q \rVert \lVert c \rVert}
= \frac{q}{\lVert q \rVert} \cdot \frac{c}{\lVert c \rVert},
$$

so normalizing the rows of both matrices first turns the same product into the
cosine table. A matrix product is the operation every backend has spent the most
effort on (blocked, vectorized, threaded, or on the GPU), which is why "a loop that
computes one inner product at a time" loses by two orders of magnitude on the same
arithmetic.

**Selecting.** Sorting a row costs $O(N \log N)$ and orders all the entries nobody
asked for. Selection finds the $k$ largest in $O(N)$ (quickselect, or a heap of
size $k$ in $O(N \log k)$), and only those $k$ are then sorted.

**Ties.** The catch is that selection routines are not stable. `torch.topk`
documents that the indices of tied elements "are not guaranteed to be stable", and
`mx.argpartition` leaves the order within each side of the partition undefined.
What *is* well defined is the set of the $k$ largest **values**. So ask the
selection for one number per row, the $k$-th best score $t$, and build the answer
from it:

- every entry with score $> t$ is in (there are fewer than $k$ of them);
- the remaining places go to entries with score $= t$, in index order. A running
  count (`cumsum`) of the tied entries along the row numbers them 1, 2, 3, ... from
  the left, so "the first `room` of them" is one comparison.

That mask has exactly $k$ true entries per row; listing them gives the right set in
ascending index order, and a stable descending sort of $k$ scores finishes.

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

`keep.nonzero()` lists the true positions in row-major order, so its column
coordinates, reshaped to `(Q, k)`, are each row's kept indices in ascending order.

The other two references differ, because the frameworks do:

- **JAX**: `jax.lax.top_k(scores, k)` returns values and indices, best first, and
  is stable by default (`is_stable=True`: equal elements keep their input order).
  It is the whole selection.
- **MLX**: `mx.topk` returns values only, and `mx.argpartition` has the tie problem.
  `mx.argsort` is stable and runs on the GPU, where a full sort of a 20,000-entry
  row costs about what a partition does, so the reference is
  `mx.argsort(-scores, axis=1)[:, :k]`. Negating instead of reversing matters: a
  reversed ascending sort would put the *higher* index first among equals.

## Common mistakes

- **A plain inner product for cosine.** Without normalization long vectors win
  every query. On a corpus of unit vectors the bug is invisible, which is how it
  ships: the embedding model is swapped for one that does not normalize.
- **Normalizing one side.** Normalizing only the corpus gives the right ranking,
  because each row of the table is scaled by one positive number, and wrong scores.
  A similarity threshold tuned against true cosines then cuts in the wrong place.
- **Ignoring `metric`.** Models trained for inner-product search put information in
  the length of the vector; normalizing it away changes the ranking.
- **Trusting the indices of a top-k.** The values are right and the scores match;
  the indices among equal scores are arbitrary, and can differ between runs, devices,
  and library versions.
- **An unstable full sort**, the default of `torch.sort` and `torch.argsort`. Same
  symptom; pass `stable=True`.
- **Not capping `k`.** `topk` raises when `k > N`. A small corpus (a new tenant, a
  filtered subset) is where this happens in production.
- **A loop over the corpus.** The same arithmetic as one product, issued as $N$
  framework calls.

## Complexity

The product is $O(QND)$ and the selection $O(QN)$, plus $O(Qk \log k)$ to order the
winners; the score table takes $O(QN)$ memory. For a large corpus, score it in
blocks and merge each block's top $k$ with the running best, which bounds the table
at `Q * block` entries.

## In the wild

- Faiss calls this index `IndexFlatIP` (inner product) and `IndexFlatL2`; it is the
  baseline its documentation measures every other index against, and the usual
  choice for a quantizer over centroids.
- Sentence Transformers' `semantic_search` does exactly this in chunks of queries
  and corpus rows, with `torch.topk` on each chunk.
- Normalizing once at indexing time and storing unit vectors makes cosine search an
  inner-product search, which is why vector databases ask whether your embeddings
  are normalized.
