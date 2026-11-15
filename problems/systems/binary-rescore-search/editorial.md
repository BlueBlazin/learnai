## The idea

**One bit per dimension.** For vectors on the unit sphere, the fraction of
coordinates in which two vectors have different signs tends to grow with the angle
between them. So the Hamming distance between sign codes is a noisy estimate of angular
distance: good enough to tell near from far, too coarse to order the nearest few.

**Packed bits.** Stored as one byte per bit the codes would save nothing over
int8. Packed, a 1024-dimensional vector is 128 bytes, and the distance is computed
without ever unpacking:

$$
d_H(a, b) = \operatorname{popcount}(a \oplus b) .
$$

XOR marks the differing positions; counting the marks is the distance. CPUs have
an instruction for the count. In a tensor framework without one, a byte has only
256 values, so a table of 256 precomputed counts turns popcount into an index
operation: `POPCOUNT[xor]` gives the count of every byte at once, and a sum over
the last axis adds the bytes of a code.

**Two stages.** The scan over codes is cheap, so let it return more than is
needed, $k \cdot \texttt{oversample}$ candidates, and spend the precise comparison
on those. Three things make the second stage precise:

- the **query is not quantized**: it is one vector, and its float values weight
  each coordinate properly;
- the **documents are int8**, four times smaller than float32 and nearly as
  informative; only the candidates' rows are read, so the int8 matrix can live on
  disk;
- the score is an inner product, which for int8 rows that are a scaled copy of
  unit vectors ranks like cosine similarity.

Oversampling is what gives rescoring something to do. With `oversample = 1` the
candidates *are* the answer set and rescoring can only reorder it. A true neighbor
that the bits ranked 25th is recovered only if the candidate list reaches rank 25.

## Reference solution (PyTorch)

```python
POPCOUNT = torch.tensor([byte.bit_count() for byte in range(256)])
BIT_VALUES = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1])


def binarize(e):
    bits = (e > 0).reshape(*e.shape[:-1], -1, 8)
    return (bits * BIT_VALUES).sum(dim=-1).to(torch.uint8)


def hamming_topk(q_bits, corpus_bits, m):
    differing = torch.bitwise_xor(corpus_bits, q_bits)
    distances = POPCOUNT[differing.long()].sum(dim=-1)
    order = torch.argsort(distances, stable=True)[:m]
    return order, distances[order]


def search(q, corpus_bits, corpus_int8, k, oversample, distance):
    candidates, _ = hamming_topk(binarize(q), corpus_bits, k * oversample)
    scores = distance(q, corpus_int8[candidates])
    order = torch.argsort(scores, descending=True, stable=True)[:k]
    return candidates[order], scores[order]
```

Slicing with `[:m]` and `[:k]` caps both counts at what exists, so `m > N` needs no
special case. In JAX and MLX the table is indexed with the uint8 tensor directly,
and the descending stable sort is an ascending sort of the negated scores.

For a large corpus the full sort in `hamming_topk` is the part to replace. Hamming
distances are small integers, so the tie rule can be folded into the key:
`distances * N + arange(N)` is distinct for every document and orders by distance,
then index; any partial selection of the `m` smallest keys is then exact.

## Common mistakes

- **Rescoring everything.** Calling `distance(q, corpus_int8)` and taking the top
  `k` gives results at least as good and reads the whole int8 matrix: the cost of
  the search the binary index was built to avoid.
- **Counting differing bytes.** `(a != b).sum()` on packed codes counts bytes, so
  a byte with eight differing bits counts like a byte with one. The ranking
  degrades quietly; nothing crashes.
- **Ranking in ascending order.** Hamming *distances* are sorted ascending, inner
  products descending. Reusing the same sort for both returns the worst candidates.
- **`>= 0` for the sign.** Exact zeros are common (ReLU outputs, zero-padded
  Matryoshka prefixes), and the corpus was binarized with `> 0`.
- **The wrong bit order.** Packing with the first coordinate in the least
  significant bit is self-consistent and incompatible with every index built by
  `numpy.packbits`.
- **Ignoring `oversample`.** Retrieving `k` candidates and "rescoring" them only
  reorders the Hamming top `k`.
- **Scoring without reordering.** Returning the first `k` candidates with their
  new scores looks like a rescored result and is the Hamming ranking.
- **Arbitrary ties in the candidate stage.** With codes of a few bytes, hundreds
  of documents share a distance; which of them become candidates must not depend
  on the selection routine.

## Complexity

The Hamming pass is $O(N \cdot D / 8)$ byte operations and reads $N D / 8$ bytes.
Rescoring is $O(k \cdot \texttt{oversample} \cdot D)$ and reads that many int8
values. Exact float search is $O(ND)$ float operations over $4ND$ bytes.

## In the wild

- The Sentence Transformers write-up on embedding quantization ("Binary and Scalar
  Embedding Quantization for Significantly Faster & Cheaper Retrieval", Hugging
  Face blog) describes this pipeline: binary codes made by thresholding normalized
  embeddings at 0 and packing with `np.packbits`, retrieval of
  `rescore_multiplier * top_k` candidates, and rescoring with the float32 query;
  its demo rescoring step loads the candidates' int8 embeddings from disk. Its
  experiments use a multiplier of 4, and report about 96% of float32 retrieval
  quality for binary codes with rescoring on the model it measures, against about
  92.5% without.
- `quantize_embeddings(..., precision="ubinary")` in Sentence Transformers is
  `np.packbits(embeddings > 0)`; `precision="binary"` is the same bytes shifted to
  int8. Its `semantic_search_faiss` and `semantic_search_usearch` rescore by
  default, with `rescore_multiplier=2`.
- On the seeded corpus of this problem the Hamming stage alone finds 61% of the
  true top 10, and four-fold oversampling with int8 rescoring finds 96%.
