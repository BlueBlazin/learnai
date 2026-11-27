Search a corpus whose vectors are stored in two forms. A compact 1-bit-per-dimension copy supports a fast first pass, and a 1-byte-per-dimension copy supports a precise second pass over the candidates that the first pass returns. Implement three functions to pack sign bits, run a Hamming search over packed codes, and rescore oversampled candidates with the full-precision query.

## Specification

```python
def binarize(e): ...
def hamming_topk(q_bits, corpus_bits, m): ...
def search(q, corpus_bits, corpus_int8, k, oversample, distance): ...
```

Let $N$ be the number of documents in the corpus and $D$ be the embedding dimension, where $D$ is a multiple of 8.

**`binarize(e)`** converts float vectors of shape `(..., D)` into packed sign bits. Each coordinate $e_j$ becomes a single bit $b_j$:

$$
b_j = \begin{cases} 1 & e_j > 0 \\ 0 & e_j \le 0 \end{cases}
$$

The function packs every 8 consecutive bits into one uint8 byte, placing the first bit of each group in the **most significant** bit position:

$$
\text{byte}_i = \sum_{j=0}^{7} b_{8i + j} \cdot 2^{\,7 - j} .
$$

This matches `numpy.packbits(e > 0, axis=-1)` and returns a uint8 tensor of shape `(..., D // 8)`.

**`hamming_topk(q_bits, corpus_bits, m)`** finds the corpus codes nearest to a query code `q_bits`. The Hamming distance between two codes is the number of bit positions where they differ across all $D$ bits. Let $M = \min(m, N)$. The function returns `(indices, distances)`, two integer tensors of shape `(M,)` containing the indices and Hamming distances of the $M$ closest corpus codes in ascending distance order. Break distance ties by corpus index, lower first. That tie-breaking rule determines both the order and which codes make the top $M$.

**`search(q, corpus_bits, corpus_int8, k, oversample, distance)`** runs a two-stage search for one float query `q` of shape `(D,)`. In the corpus, `corpus_int8[i]` is document $i$'s float vector scaled by a constant and rounded to int8, and `corpus_bits[i]` is `binarize` applied to that same float vector. The `distance` argument is a kit `CountingDistance` with metric `"dot"`. Calling `distance(q, rows)` takes the float query `(D,)` and a numeric tensor `rows` of shape `(M, D)`, returns their float32 inner products of shape `(M,)` (higher is more similar), and adds $M$ to `distance.counters.distances`.

The search runs in three steps:

1. **Candidates.** Binarize `q` and call `hamming_topk` to retrieve the $\min(k \cdot \texttt{oversample}, N)$ nearest codes in `corpus_bits`. Keep the returned indices in the order `hamming_topk` produces.
2. **Rescore.** Score every candidate, and only the candidates, by calling `distance(q, rows)` where `rows` holds those candidates' rows from `corpus_int8`. Pass the float query `q`, not its binary code.
3. **Rank.** Return `(indices, scores)` for the $\min(k, N)$ highest-scoring candidates in descending score order. Keep candidates with equal scores in their candidate order from step 1.

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

Because zero maps to `0`, the signs produce the bits `1 0 0 1 0 1 0 0`. In binary, `0b10010100` equals 148.

Now consider a corpus of five documents with $D = 8$, so each code is one byte:

```python
corpus_int8 = [
    [10, 10, 10, 10, -10, -10, -10, -10],   # bits 11110000 = 240
    [100, -5, 5, 5, -5, -5, -5, -5],        # bits 10110000 = 176
    [-100, 5, 5, 5, -5, -5, -5, -5],        # bits 01110000 = 112
    [-10, -10, -10, -10, 10, 10, 10, 10],   # bits 00001111 =  15
    [50, 50, -5, -5, 5, -5, -5, -5],        # bits 11001000 = 200
]
corpus_bits = [[240], [176], [112], [15], [200]]
q = [0.75, 0.125, 0.125, 0.125, -0.125, -0.125, -0.125, -0.125]  # bits 11110000 = 240

hamming_topk([240], corpus_bits, 3)  # indices [0, 1, 2], distances [0, 1, 1]
```

The Hamming distances from `q` to the five codes are `0, 1, 1, 8, 3`. Documents 1 and 2 tie at distance 1, so the lower index comes first.

```python
search(q, corpus_bits, corpus_int8, k=1, oversample=1, distance)   # ([0], [16.25])
search(q, corpus_bits, corpus_int8, k=1, oversample=2, distance)   # ([1], [78.125])
search(q, corpus_bits, corpus_int8, k=2, oversample=2, distance)   # ([1, 4], [78.125, 43.75])
```

With `oversample=1`, the only candidate is document 0, so the search returns it with score `16.25`. With `oversample=2`, the candidates are documents 0 and 1, whose inner products with `q` are `16.25` and `78.125`. Document 1 wins even though its binary code differs by one bit, because its first coordinate is ten times larger where `q` places most of its weight. With `k=2` and `oversample=2`, the four candidates `0, 1, 2, 4` score `16.25, 78.125, -70.625, 43.75`, so documents 1 and 4 take the top two spots. Across these three calls, `distance` evaluates 1, 2, and 4 inner products, never all 5.

## Constraints

- $8 \le D \le 1024$ with $D$ a multiple of 8, and $1 \le N \le 20{,}000$.
- $k \ge 1$, `oversample` $\ge 1$, and $m \ge 1$. Both `m` and `k * oversample` may exceed `N`.
- Codes, indices, and Hamming distances are compared exactly. Scores are compared with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- **Cost.** After `search`, `distance.counters.distances` must be at most $\min(k \cdot \texttt{oversample}, N)$. The rescoring pass may touch only the candidates, so scoring the entire corpus fails this check.
- **Recall.** On the seeded test corpus (2,000 unit vectors of 64 dimensions in 100 clusters), the top 10 results of `search` with `oversample=4` must contain at least 90% of the true float top 10 by inner product, averaged over 20 queries. The reference solution reaches 96%, while the Hamming stage alone reaches 61%.
- No Python loop over corpus rows is needed.

## Why this matters

A 1024-dimensional float32 embedding takes 4 KB, so 100 million embeddings require 400 GB of RAM. Storing 1 bit per dimension cuts memory by 32x, and comparing two binary codes takes only an XOR and a population count. That keeps the index in memory and makes full scans fast, but sign bits throw away coordinate magnitudes.

Two-stage search recovers most of the lost accuracy. The binary scan proposes a small multiple of `k` candidates, and the second pass rescores only those candidates using a higher-precision int8 copy (often stored on disk or in cheaper memory) against the full-precision float query. Keeping the expensive pass limited to a few dozen vectors per query is what makes the pipeline practical, and it is why the cost bound is enforced.
