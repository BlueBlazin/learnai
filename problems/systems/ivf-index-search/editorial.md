## The idea

The centroids cut the space into cells: each point belongs to the cell of its
nearest centroid (a Voronoi partition). The index stores, for every cell, the list
of corpus vectors inside it. A query's nearest neighbors are probably in the cell
the query falls in, or in a cell next to it, so the search ranks the cells by the
distance from the query to their centroids and scans the first `nprobe`.

**Cost.** The coarse step costs $C$ distances and the scan costs the sizes of the
probed lists, about $\texttt{nprobe} \cdot N / C$ when the lists are balanced. With
$C \approx \sqrt{N}$ and a few probes, that is a few times $\sqrt{N}$ instead of
$N$.

**Why it can miss, and why more probes never hurt.** A cell boundary can pass
between the query and its true neighbor, as in the worked example, where the
neighbor was assigned to the other side of a tie. Raising `nprobe` adds lists to
the scan and never removes one, because the probe order does not depend on
`nprobe`. So the candidate set only grows. A true top-$k$ neighbor that is among
the candidates is always returned (at most $k - 1$ vectors anywhere are closer to
the query than it is), so every neighbor found with fewer probes is still found
with more: recall is monotone. At `nprobe = C` every vector is a candidate and the
search is exact.

**Layout.** Lists of Python objects would work and would be slow to scan. Storing
the vectors grouped by list (a stable sort by list number) makes each list a
contiguous block that one slice, or one gather, reads; `offsets` is the cumulative
sum of the list sizes, and `ids` remembers where each stored vector came from. It
is the same idea as the CSR layout of a sparse matrix.

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

Three stable sorts carry the three tie rules: ascending ids within a list, lower
centroid id among equally near centroids, scan order among equally near
candidates. JAX is a direct translation (`jnp.bincount(..., minlength=C)`;
`jnp.argsort` is stable). MLX has no `bincount`, so the sizes are counted with a
comparison against `arange(C)`, and the assignment takes the first column of a
stable row sort so that the tie rule does not rest on what `argmin` does with
equal values.

The `(N, C, D)` intermediate of the build is fine at these sizes; for a real corpus
compute the table in blocks, or as
$\lVert x \rVert^2 - 2\, x \cdot c + \lVert c \rVert^2$ with one matrix product.

## Common mistakes

- **Probing lists by id.** Scanning lists `0 .. nprobe - 1` returns well-formed
  results from the wrong part of the space. Recall is that of a random sample.
- **Losing empty lists.** `bincount` without `minlength` (or a `unique`-based
  count) is one entry short for every trailing list without members, so `offsets`
  has the wrong length and the probe of such a list raises. Empty lists are normal:
  centroids trained on a sample, or on last month's data, attract nothing.
- **Padding short results.** When the probed lists hold fewer than `k` vectors,
  the answer is shorter than `k`. Filling it up by repeating entries hands the
  caller duplicate documents. (Faiss fills the missing slots with the label $-1$
  instead; here the result is simply shorter.)
- **Scanning everything, then filtering.** Computing all $N$ distances and masking
  to the probed lists gives the same answer at the cost of the exact scan.
- **Returning storage positions.** `positions[best]` indexes the grouped storage;
  the caller needs corpus indices, `ids[positions[best]]`. On a corpus that happens
  to be sorted by cluster the two coincide.
- **Assigning by inner product.** The nearest centroid in L2 is not the centroid
  with the largest inner product unless every vector has the same length.
- **Unsorted lists.** An unstable sort by list number groups correctly and leaves
  each list in arbitrary order, which changes the tie order of every search.

## Complexity

Build: $O(NCD)$ for the assignment and $O(N \log N)$ for the sort. Search:
$O(CD)$ for the coarse step, $O(MD)$ for a scan of $M$ candidates, and
$O(M \log M)$ for the final sort (a partial selection of the $k$ smallest brings
that to $O(M)$).

## In the wild

- Faiss's `IndexIVFFlat` is this index: its documentation describes the space as
  partitioned into `nlist` cells, the vectors stored in `nlist` inverted lists, a
  *quantizer* (usually a flat index over the centroids) that picks the lists to
  visit, and `nprobe`, set at query time, as the number of lists searched. It
  notes that `nprobe / nlist` underestimates the fraction of the corpus scanned,
  because the lists are not equally long.
- The inverted file comes from text retrieval; "Video Google: A Text Retrieval
  Approach to Object Matching in Videos" (ICCV 2003) carried it over to quantized
  descriptors, with the centroids playing the role of words.
- IVF-PQ keeps the same lists and stores compressed codes in them; the search
  logic of this problem is unchanged, only the scan computes approximate distances.
