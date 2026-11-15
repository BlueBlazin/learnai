Search a corpus whose vectors are stored twice: as one bit per dimension, for a fast
and coarse first pass, and as one byte per dimension, for a precise second pass over
the few candidates the first pass returns. Implement the three pieces: packing sign
bits, the Hamming search over packed codes, and the search that retrieves more
candidates than it needs and rescores them with the full-precision query.

## Specification

```python
def binarize(e): ...
def hamming_topk(q_bits, corpus_bits, m): ...
def search(q, corpus_bits, corpus_int8, k, oversample, distance): ...
```

**`binarize(e)`** turns float vectors into packed sign bits. `e` has shape
`(..., D)` with `D` a multiple of 8. Coordinate $j$ of a vector becomes the bit

$$
b_j = \begin{cases} 1 & e_j > 0 \\ 0 & e_j \le 0 \end{cases}
$$

and the bits are packed eight to a byte, the first of each group in the **most
significant** bit:

$$
\text{byte}_i = \sum_{j=0}^{7} b_{8i + j} \cdot 2^{\,7 - j} .
$$

It returns a uint8 tensor of shape `(..., D // 8)`. This is `numpy.packbits(e > 0,
axis=-1)`.

**`hamming_topk(q_bits, corpus_bits, m)`** finds the codes nearest to a query code.
The Hamming distance of two codes is the number of bit positions in which they
differ, over all $D$ bits. With $M = \min(m, N)$ it returns the tuple
`(indices, distances)`, two integer tensors of shape `(M,)`: the indices of the $M$
corpus codes with the smallest distance to `q_bits` and those distances, in
ascending order of distance. Equal distances are ordered by index, lower first
(this decides both the order and which codes are among the $M$).

**`search(q, corpus_bits, corpus_int8, k, oversample, distance)`** is the two-stage
search for one float query `q` of shape `(D,)`:

1. **Candidates.** Binarize `q` and take the $\min(k \cdot \texttt{oversample}, N)$
   nearest codes of `corpus_bits`, with `hamming_topk`. Their indices, in the order
   `hamming_topk` returns them, are the candidates.
2. **Rescore.** Score every candidate, and only the candidates, with
   `distance(q, rows)`, where `rows` are the candidates' rows of `corpus_int8`. The
   query is the float vector, not its bits.
3. **Rank.** Return the tuple `(indices, scores)`: the corpus indices of the
   $\min(k, N)$ candidates with the highest scores, best first, and their scores.
   Candidates with equal scores stay in candidate order.

`corpus_int8[i]` is document $i$'s float vector scaled by a constant and rounded to
int8, and `corpus_bits[i]` is `binarize` of the same float vector. `distance` is a
kit `CountingDistance` with the metric `"dot"`: `distance(q, rows)` takes the float
query `(D,)` and a tensor of rows `(M, D)` of any numeric dtype, returns their inner
products as a float32 tensor of shape `(M,)` (higher is more similar), and adds $M$
to `distance.counters.distances`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `e` | `(..., D)` | float32 |
| `binarize(e)` | `(..., D // 8)` | uint8 |
| `q_bits` | `(D // 8,)` | uint8 |
| `corpus_bits` | `(N, D // 8)` | uint8 |
| `hamming_topk(...)` | `(min(m, N),)` twice | integer |
| `q` | `(D,)` | float32 |
| `corpus_int8` | `(N, D)` | int8 |
| `search(...)` | `(min(k, N),)` twice | integer, float32 |

## Example

```python
binarize([0.5, -1.0, 0.0, 2.0, -0.1, 3.0, 0.0, -4.0])  # [148]
```

The signs give the bits `1 0 0 1 0 1 0 0` (zero is not positive), and
`0b10010100` is 148.

A corpus of five documents with $D = 8$, one byte of code each:

```python
corpus_int8 = [
    [10, 10, 10, 10, -10, -10, -10, -10],  # bits 11110000 = 240
    [100, -5, 5, 5, -5, -5, -5, -5],  # bits 10110000 = 176
    [-100, 5, 5, 5, -5, -5, -5, -5],  # bits 01110000 = 112
    [-10, -10, -10, -10, 10, 10, 10, 10],  # bits 00001111 =  15
    [50, 50, -5, -5, 5, -5, -5, -5],
]  # bits 11001000 = 200
corpus_bits = [[240], [176], [112], [15], [200]]
q = [0.75, 0.125, 0.125, 0.125, -0.125, -0.125, -0.125, -0.125]  # bits 11110000 = 240

hamming_topk([240], corpus_bits, 3)  # indices [0, 1, 2], distances [0, 1, 1]
```

The distances to the five codes are `0, 1, 1, 8, 3`. Documents 1 and 2 tie at
distance 1, and the lower index comes first.

```python
search(q, corpus_bits, corpus_int8, k=1, oversample=1, distance)   # ([0], [16.25])
search(q, corpus_bits, corpus_int8, k=1, oversample=2, distance)   # ([1], [78.125])
search(q, corpus_bits, corpus_int8, k=2, oversample=2, distance)   # ([1, 4], [78.125, 43.75])
```

With `oversample=1` the only candidate is document 0, the nearest code, and the
search returns it. With `oversample=2` the candidates are documents 0 and 1; their
inner products with `q` are 16.25 and 78.125, so document 1 wins: its code is one
bit away, but its first coordinate, where the query has most of its weight, is ten
times larger. For `k=2` the four candidates `0, 1, 2, 4` score
`16.25, 78.125, -70.625, 43.75`. `distance` computed 1, 2, and 4 inner products in
the three calls, never 5.

## Constraints

- $8 \le D \le 1024$, a multiple of 8; $1 \le N \le 20{,}000$; $k \ge 1$,
  `oversample` $\ge 1$, $m \ge 1$; `m` and `k * oversample` may exceed `N`.
- Codes, indices, and Hamming distances are compared exactly; scores to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- **Cost.** `distance.counters.distances` is at most $\min(k \cdot
  \texttt{oversample}, N)$ after `search`: the full-precision pass touches the
  candidates and nothing else. Scoring the whole corpus exceeds it.
- **Recall.** On the seeded corpus of the tests (2,000 unit vectors of 64
  dimensions in 100 clusters), the top 10 of `search` with `oversample=4` contain
  at least 90% of the true top 10 by inner product of the float vectors, averaged
  over 20 queries (the reference reaches 96%). The Hamming stage alone reaches 61%
  there.
- No Python loop over corpus rows is needed.

## Why this matters

A float32 embedding of 1024 dimensions is 4 KB; a hundred million of them are
400 GB of RAM. One bit per dimension is 32 times smaller, and comparing two codes
is an XOR and a population count, so the index fits in memory and a scan is fast.
The price is precision: the sign of a coordinate says little about its size. The
standard remedy is the one you implement here: let the bits propose a few times
more candidates than needed, then score those with a better copy of the vectors,
int8 here, kept on disk or in cheaper memory, against the query at full precision.
The recall lost by the bits is mostly recovered, and the expensive pass stays at a
few dozen vectors per query, which is the reason the cost limit exists.
