Build an inverted-file (IVF) index and search it. The index groups the corpus by
nearest centroid; a search compares the query with the centroids first, and then
scans only the few groups whose centroids are closest. It is the simplest index that
does less work than an exact scan, and the price is that it can miss: a neighbor
that sits in a group you did not open is never seen.

## Specification

```python
def build_ivf(corpus, centroids): ...
def search(q, index, k, nprobe, distance): ...
```

All distances are **squared L2 distances**, $\lVert a - b \rVert_2^2$; smaller is
closer. The centroids are given (they would come from k-means on a sample of the
corpus); there are $C$ of them, and with them $C$ **inverted lists**, numbered like
the centroids.

**`build_ivf(corpus, centroids)`** assigns every corpus vector to the list of its
nearest centroid. If two centroids are equally near, the vector goes to the one with
the lower id. It returns the tuple `(centroids, offsets, ids, vectors)`:

- `centroids`: the `(C, D)` tensor it was given.
- `ids`: an integer tensor of shape `(N,)`, the corpus indices **grouped by list**:
  first the members of list 0, then those of list 1, and so on. Within a list the
  indices are in ascending order.
- `offsets`: an integer tensor of shape `(C + 1,)`. List $j$ occupies positions
  `offsets[j]` to `offsets[j + 1] - 1` of `ids`. So `offsets[0] = 0`,
  `offsets[C] = N`, and an empty list has `offsets[j] == offsets[j + 1]`.
- `vectors`: the `(N, D)` tensor `corpus[ids]`: the corpus reordered so that each
  list is one contiguous block.

![The index of the worked example. `ids` and `vectors` hold the corpus grouped by list, and `offsets` marks where each list begins and ends; a search that probes list 1 reads positions 3 to 5 and nothing else.](assets/list-layout.svg)

**`search(q, index, k, nprobe, distance)`** finds near neighbors of one query `q`
of shape `(D,)`. `distance` is a kit `CountingDistance` with the metric `"l2"`:
`distance(q, rows)` takes `q` and a tensor of rows `(M, D)`, returns their squared
L2 distances to `q` as a float32 tensor of shape `(M,)`, and adds $M$ to
`distance.counters.distances`. `rows` may have zero rows. **Every distance the
search uses must come from `distance`.**

1. **Coarse.** Compute the distance from `q` to every centroid. The **probed
   lists** are the $\min(\texttt{nprobe}, C)$ lists whose centroids are nearest,
   nearest first; equally near centroids are ordered by id, lower first.
2. **Scan.** The candidates are the members of the probed lists, in **scan order**:
   list by list in probe order, and within a list in stored order. Compute the
   distance from `q` to each candidate, and to no other vector.
3. **Select.** Return the tuple `(ids, distances)`: the corpus indices of the
   $\min(k, \text{number of candidates})$ nearest candidates and their distances,
   nearest first. Candidates at equal distance stay in scan order.

If the probed lists hold fewer than `k` vectors, the result is shorter than `k`; if
they hold none, both tensors have shape `(0,)`. No id appears twice.

## Shapes

| name | shape | dtype |
|---|---|---|
| `corpus` | `(N, D)` | float32 |
| `centroids` | `(C, D)` | float32 |
| `offsets` | `(C + 1,)` | integer |
| `ids` (index) | `(N,)` | integer |
| `vectors` | `(N, D)` | float32 |
| `q` | `(D,)` | float32 |
| `search(...)` | `(K,)` twice, $K = \min(k, \text{candidates})$ | integer, float32 |

## Example

Three centroids and six vectors in the plane:

```python
centroids = [[0, 0], [4, 0], [0, 4]]
corpus = [[1, 0], [4, 1], [0, 1], [2, 0], [5, 0], [3, -1]]

build_ivf(corpus, centroids)
# offsets [0, 3, 6, 6]
# ids     [0, 2, 3, 1, 4, 5]
# vectors [[1, 0], [0, 1], [2, 0], [4, 1], [5, 0], [3, -1]]
```

Vectors 0 and 2 are nearest to centroid 0. Vector 3, `[2, 0]`, is at distance 4
from both centroid 0 and centroid 1, and goes to list 0, the lower id. Vectors 1,
4, and 5 belong to list 1. Nothing is nearest to centroid 2: list 2 is empty.

```python
q = [2.5, 0]
search(q, index, k=2, nprobe=1, distance)   # ids [5, 1], distances [1.25, 3.25]
search(q, index, k=2, nprobe=2, distance)   # ids [3, 5], distances [0.25, 1.25]
search(q, index, k=10, nprobe=1, distance)  # ids [5, 1, 4], distances [1.25, 3.25, 6.25]
```

The centroid distances are `6.25, 2.25, 22.25`, so the probe order is list 1, list
0, list 2. With `nprobe=1` only list 1 is scanned: 3 centroid distances and 3
vector distances, 6 in all. The true nearest neighbor, vector 3 at distance 0.25,
is in list 0 and is missed. With `nprobe=2` lists 1 and 0 are scanned (9
distances) and it is found. With `nprobe=3` the empty list 2 is probed as well,
which changes neither the result nor the count. With `k=10` and one probe there
are only three candidates, so three results.

## Constraints

- $1 \le N \le 20{,}000$, $1 \le C \le 256$, $1 \le D \le 64$, $k \ge 1$,
  `nprobe` $\ge 1$; `nprobe` may exceed $C$ and `k` may exceed the number of
  candidates. Lists may be empty.
- `offsets` and `ids` are compared exactly, and so are the ids a search returns;
  vectors and distances to a relative tolerance of `1e-4` and an absolute tolerance
  of `1e-5`. In the test corpora no vector is within `1e-3` of being equally near
  two centroids, except in integer examples where the tie is exact.
- **Cost.** After `search`, `distance.counters.distances` is exactly

  $$
  C + \sum_{j \,\in\, \text{probed lists}} \lvert \text{list } j \rvert .
  $$

  Scanning lists that are not probed gives more; computing distances without
  `distance` gives less. Both fail. `build_ivf` computes its own distances and is
  not counted.
- **Properties.** With `nprobe` $\ge C$ the result equals the exact nearest
  neighbors of a full scan. Recall (the fraction of the true `k` nearest that are
  returned) never decreases when `nprobe` grows.
- A Python loop over the probed lists is fine; no loop over vectors is needed.

## Why this matters

An exact scan of a hundred million vectors per query is not a service. IVF is the
first thing vector databases reach for: with $C$ lists and `nprobe` probes a query
touches about `nprobe / C` of the corpus, and `nprobe` is a dial that trades recall
for latency at query time, without rebuilding anything. It is also the frame that
the heavier indexes are built in: the lists hold compressed codes instead of
vectors (IVF-PQ), or binary codes with rescoring. The failures are quiet ones.
Probing the wrong lists, dropping an empty list, or returning positions instead of
ids all produce results of the right shape, and only a recall measurement against
an exact scan shows that the index is broken.
