## The idea

The centroids partition the vector space into Voronoi cells, where each cell contains all points closest to its centroid. The index stores one inverted list per cell holding the corpus vectors inside it. Because a query's nearest neighbors usually lie in the query's own cell or an adjacent one, a search ranks the centroids by distance to the query and scans only the top `nprobe` lists.

**Cost.** The coarse step evaluates $C$ centroid distances, and the scan evaluates the vectors in the probed lists, roughly $\texttt{nprobe} \cdot N / C$ when lists are balanced. Choosing $C \approx \sqrt{N}$ with a small `nprobe` reduces the work from $N$ distances to a small multiple of $\sqrt{N}$.

**Why it can miss, and why more probes never hurt.** A cell boundary can separate a query from its true nearest neighbor, as in the worked example where vector 3 fell on the other side of a tie. Increasing `nprobe` only appends lists to the scan and never removes any, because the probe order does not depend on `nprobe`. As a result, the candidate set only grows. Once a true top-$k$ neighbor enters the candidate set, at most $k - 1$ vectors in the entire corpus are closer to the query than it is, so it is guaranteed to appear in the returned top $k$. Recall is therefore monotone in `nprobe`, and at `nprobe = C` every vector is scanned so the search is exact.

**Layout.** Storing each list as a Python list of vectors would be slow to scan. Grouping all vectors by list via a stable sort makes every list a single contiguous block that can be read with one slice or gather. The `offsets` array holds the cumulative sum of list sizes, and `ids` maps each storage position back to its original corpus index. This is the same layout as Compressed Sparse Row (CSR) format for sparse matrices.

## Reference solution (PyTorch)

```python
def build_ivf(corpus, centroids):
    gaps = corpus[:, None, :] - centroids[None, :, :]
    assigned = (gaps * gaps).sum(dim=-1).argmin(dim=1)
    ids = torch.argsort(assigned, stable=True)
    sizes = torch.bincount(assigned, minlength=centroids.shape[0])
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(dim=0)])
    return centroids, offsets, ids, corpus[ids]


def search(q, index, k, nprobe, distance):
    centroids, offsets, ids, vectors = index
    to_centroids = distance(q, centroids)
    probe = torch.argsort(to_centroids, stable=True)[:nprobe].tolist()

    bounds = offsets.tolist()
    positions = torch.cat([torch.arange(bounds[j], bounds[j + 1]) for j in probe])

    distances = distance(q, vectors[positions])
    best = torch.argsort(distances, stable=True)[:k]
    return ids[positions[best]], distances[best]
```

Three stable sorts enforce the three tie-breaking rules: ascending corpus ids within each list, lower centroid id among equally near centroids, and scan order among equally near candidates. JAX translates directly with `jnp.bincount(..., minlength=C)` and `jnp.argsort` (which is stable). MLX has no `bincount`, so you can count list sizes by comparing `assigned` against `arange(C)` and take the first column of a stable row sort instead of relying on `argmin` tie behavior.

Materializing the `(N, C, D)` difference tensor during `build_ivf` is fine at these sizes. For a production corpus, compute the squared distance between each vector $x$ and centroid $c$ in blocks or via $\lVert x \rVert^2 - 2\, x \cdot c + \lVert c \rVert^2$ using matrix multiplication.

## Common mistakes

- **Probing lists by id.** Scanning lists `0` through `nprobe - 1` produces well-formed outputs from the wrong region of space, giving recall no better than a random sample.
- **Losing empty lists.** Calling `bincount` without `minlength`, or counting via `unique`, omits trailing empty lists. That makes `offsets` too short and causes out-of-bounds errors when probing those lists. Empty lists happen routinely when centroids are trained on a sample or on stale data.
- **Padding short results.** When the probed lists hold fewer than `k` vectors, return fewer than `k` results rather than repeating entries to pad the output. (Faiss pads missing slots with the sentinel label $-1$, whereas this specification returns a shorter tensor.)
- **Scanning everything, then filtering.** Computing all $N$ distances and masking to the probed lists returns the right answer at the cost of a full scan, failing the cost check.
- **Returning storage positions.** `positions[best]` indexes into the reordered `vectors` array rather than the original corpus. You must map back through `ids[positions[best]]`. If a test corpus happens to be pre-sorted by cluster, this bug hides until an unsorted corpus is tested.
- **Assigning by inner product.** The centroid with the largest inner product is not the nearest centroid in L2 distance unless all vectors share the same norm.
- **Unsorted lists.** An unstable sort by list number groups vectors into the right lists but scrambles their order within each list, breaking tie order during search.

## Complexity

Building the index takes $O(NCD)$ time for centroid assignment and $O(N \log N)$ time for sorting. Searching takes $O(CD)$ time for the coarse step, $O(MD)$ time to scan $M$ candidates, and $O(M \log M)$ time to sort them (which drops to $O(M)$ with partial top-$k$ selection).

## In the wild

- Faiss's `IndexIVFFlat` implements this exact index. It partitions the space into `nlist` cells with `nlist` inverted lists, uses a *quantizer* (typically a flat index over the centroids) to pick which lists to visit, and exposes `nprobe` as a query-time parameter. The Faiss documentation also notes that `nprobe / nlist` underestimates the actual fraction of the corpus scanned when list sizes are unbalanced.
- The inverted-file structure originated in text retrieval. "Video Google: A Text Retrieval Approach to Object Matching in Videos" (ICCV 2003) adapted it to quantized image descriptors, using centroids as visual words.
- IVF-PQ keeps the same inverted lists and search flow, replacing raw vectors in the lists with compressed product-quantization codes so that the scan computes approximate distances.
