Build an inverted-file (IVF) index and search it. The index partitions a corpus by nearest centroid so that a search compares the query against the centroids first and scans only the closest partitions. That skips most of the corpus, though any true neighbor in a partition you do not visit will be missed.

## Specification

```python
def build_ivf(corpus, centroids): ...
def search(q, index, k, nprobe, distance): ...
```

Let $N$ be the number of corpus vectors, $C$ be the number of centroids, and $D$ be the vector dimension. All distances are **squared L2 distances**, $\lVert a - b \rVert_2^2$, where smaller means closer. The $C$ centroids are given (in practice they come from k-means on a sample of the corpus), and they define $C$ **inverted lists** numbered `0` through `C - 1`.

**`build_ivf(corpus, centroids)`** assigns each corpus vector to the inverted list of its nearest centroid. If two centroids tie for nearest, assign the vector to the centroid with the lower id. Return the tuple `(centroids, offsets, ids, vectors)`:

- `centroids`: the `(C, D)` tensor passed in.
- `ids`: an integer tensor of shape `(N,)` holding corpus indices grouped by list, starting with list 0, then list 1, and so on. Within each list, indices appear in ascending order.
- `offsets`: an integer tensor of shape `(C + 1,)` marking list boundaries in `ids`. List $j$ spans positions `offsets[j]` through `offsets[j + 1] - 1`. Thus `offsets[0] = 0`, `offsets[C] = N`, and an empty list satisfies `offsets[j] == offsets[j + 1]`.
- `vectors`: the `(N, D)` tensor `corpus[ids]`, which reorders the corpus so that each list forms one contiguous block.

![The index of the worked example. A search that probes list 1 reads positions 3 to 5 and nothing else.](assets/list-layout.svg)

**`search(q, index, k, nprobe, distance)`** finds approximate nearest neighbors for a single query `q` of shape `(D,)`. The `distance` argument is a kit `CountingDistance` with metric `"l2"`. Calling `distance(q, rows)` takes `q` and a tensor `rows` of shape `(M, D)` (which may have zero rows), returns their float32 squared L2 distances to `q` of shape `(M,)`, and adds $M$ to `distance.counters.distances`. **Every distance computed during `search` must come from `distance`.**

The search runs in three steps:

1. **Coarse.** Compute the distance from `q` to every centroid. Select the $\min(\texttt{nprobe}, C)$ lists with the nearest centroids, ordered nearest first. Break centroid distance ties by centroid id, lower first. These are the **probed lists**.
2. **Scan.** Collect all members of the probed lists in **scan order**, meaning list by list in probe order, and within each list in stored order. Compute the distance from `q` to each candidate, and to no other vector.
3. **Select.** Return `(ids, distances)` containing the corpus indices and distances of the $\min(k, \text{number of candidates})$ nearest candidates, ordered nearest first. Keep candidates with equal distances in scan order.

If the probed lists contain fewer than `k` vectors, return all of them. If they contain no vectors, return two empty tensors of shape `(0,)`. No corpus id may appear twice.

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

Consider three centroids and six 2D vectors:

```python
centroids = [[0, 0], [4, 0], [0, 4]]
corpus = [[1, 0], [4, 1], [0, 1], [2, 0], [5, 0], [3, -1]]

build_ivf(corpus, centroids)
# offsets [0, 3, 6, 6]
# ids     [0, 2, 3, 1, 4, 5]
# vectors [[1, 0], [0, 1], [2, 0], [4, 1], [5, 0], [3, -1]]
```

Vectors 0 and 2 are nearest to centroid 0. Vector 3 (`[2, 0]`) has squared distance 4 to both centroid 0 and centroid 1, so the tie-breaking rule assigns it to list 0. Vectors 1, 4, and 5 belong to list 1. No vector is nearest to centroid 2, so list 2 is empty.

```python
q = [2.5, 0]
search(q, index, k=2, nprobe=1, distance)   # ids [5, 1], distances [1.25, 3.25]
search(q, index, k=2, nprobe=2, distance)   # ids [3, 5], distances [0.25, 1.25]
search(q, index, k=10, nprobe=1, distance)  # ids [5, 1, 4], distances [1.25, 3.25, 6.25]
```

The squared distances from `q` to the three centroids are `6.25, 2.25, 22.25`, giving probe order list 1, list 0, list 2. With `nprobe=1`, only list 1 is scanned, which takes 3 centroid distances plus 3 vector distances for 6 in total. The true nearest neighbor, vector 3 at distance `0.25`, sits in list 0 and is missed. With `nprobe=2`, lists 1 and 0 are scanned (9 distances total), and vector 3 is found. Probing with `nprobe=3` also visits empty list 2, which changes neither the result nor the distance count. With `k=10` and `nprobe=1`, only three candidates exist, so three results are returned.

## Constraints

- $1 \le N \le 20{,}000$, $1 \le C \le 256$, and $1 \le D \le 64$.
- $k \ge 1$ and `nprobe` $\ge 1$. `nprobe` may exceed $C$, `k` may exceed the number of candidates, and lists may be empty.
- `offsets`, index `ids`, and returned search ids are compared exactly. Vectors and distances are compared with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. In the test corpora, no vector is within `1e-3` of tying between two centroids, except in integer examples where ties are exact.
- **Cost.** After `search`, `distance.counters.distances` must equal

  $$
  C + \sum_{j \,\in\, \text{probed lists}} \lvert \text{list } j \rvert .
  $$

  Scanning unprobed lists exceeds this count, and computing distances without `distance` falls below it. Both fail. `build_ivf` computes its own distances and is not counted.
- **Properties.** When `nprobe` $\ge C$, the result matches an exact full scan. Recall (the fraction of the true `k` nearest neighbors returned) never decreases as `nprobe` increases.
- A Python loop over the probed lists is fine, and no loop over vectors is needed.

## Why this matters

Scanning 100 million vectors on every query is too slow for a live service. IVF is the standard first step in vector databases. With $C$ lists and `nprobe` probes, each query scans roughly `nprobe / C` of the corpus, giving you a query-time dial between recall and latency without rebuilding the index. IVF also serves as the outer structure for compressed indexes such as IVF-PQ and binary-rescoring pipelines, where each list stores compact codes instead of full vectors.

Bugs in an IVF index tend to fail silently. Probing the wrong lists, dropping an empty list, or returning storage positions instead of corpus ids all produce outputs of the expected shape. You only catch them by checking recall against an exact scan.
