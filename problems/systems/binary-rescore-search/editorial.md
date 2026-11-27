## The idea

**One bit per dimension.** For vectors on the unit sphere, the fraction of coordinates where two vectors have opposite signs tends to increase with the angle between them. The Hamming distance between sign codes is therefore a noisy estimate of angular distance. It is accurate enough to separate near neighbors from distant ones, but too coarse to rank the closest few reliably.

**Packed bits.** Storing one bit per byte would save no memory compared to int8. Packing 8 bits per byte shrinks a 1024-dimensional vector to 128 bytes and lets you compute distances without unpacking. For two packed codes $a$ and $b$, their Hamming distance is

$$
d_H(a, b) = \operatorname{popcount}(a \oplus b) ,
$$

where $\oplus$ is bitwise XOR. XOR marks every differing bit position with a `1`, and counting those `1` bits gives the Hamming distance. CPUs have a dedicated popcount instruction. In a tensor framework without one, a byte has only 256 possible values, so a 256-entry lookup table turns popcount into an indexing operation. Looking up `POPCOUNT[xor]` counts the set bits in every byte at once, and summing over the last axis produces the distance for each code.

**Two stages.** Because the binary scan is cheap, it can retrieve $k \cdot \texttt{oversample}$ candidates and hand them off to a precise second pass. Three details make rescoring accurate:

- **The query stays in float32.** There is only one query vector, so keeping its full precision properly weights each coordinate.
- **The documents are stored in int8.** Int8 vectors use four times less memory than float32 while preserving most of the magnitude information. Because only the candidates' rows are read, the int8 matrix can live on disk.
- **The score is an inner product.** For int8 rows produced by scaling unit vectors, inner product ranks documents the same way as cosine similarity.

Oversampling is what makes rescoring effective. With `oversample = 1`, the candidates already are the final `k` results, so rescoring can only reorder them. A true neighbor ranked 25th by Hamming distance can be recovered only if the candidate list includes at least 25 items.

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

Slicing with `[:m]` and `[:k]` automatically caps both counts at the available length, so `m > N` needs no special branch. In JAX and MLX, you can index `POPCOUNT` with the uint8 tensor directly, and you can perform a descending stable sort by sorting the negated scores in ascending order.

For a very large corpus, the full sort in `hamming_topk` is the first bottleneck to replace. Because Hamming distances are small integers, you can fold the tie-breaking rule into the sort key with `distances * N + arange(N)`. Every document gets a distinct key ordered by distance and then by index, so any partial top-`m` selection becomes exact.

## Common mistakes

- **Rescoring everything.** Calling `distance(q, corpus_int8)` and taking the top `k` produces equally good or better results, but it reads the entire int8 matrix. That defeats the purpose of the binary index and violates the cost limit.
- **Counting differing bytes.** Running `(a != b).sum()` on packed codes counts differing bytes rather than bits, treating a byte with 8 differing bits the same as one with 1 differing bit. Nothing crashes, but ranking quality degrades silently.
- **Ranking in ascending order.** Hamming distances are sorted ascending, while inner products are sorted descending. Reusing the same sort order for both returns the worst candidates.
- **Using `>= 0` for the sign.** Exact zeros are common in ReLU outputs and zero-padded Matryoshka prefixes, and the corpus codes are binarized with `> 0`.
- **Packing in the wrong bit order.** Placing the first coordinate in the least significant bit is internally consistent, but incompatible with any index built by `numpy.packbits`.
- **Ignoring `oversample`.** Retrieving only `k` candidates and rescoring them can only reorder the Hamming top `k`.
- **Scoring without reordering.** Returning the first `k` candidates with their rescored values looks like a rescored result, but preserves the Hamming ranking.
- **Arbitrary ties in the candidate stage.** With short binary codes, hundreds of documents can share the same Hamming distance. Which documents enter the candidate set must follow the index tie-breaking rule rather than depend on an unstable selection routine.

## Complexity

The Hamming pass performs $O(N \cdot D / 8)$ byte operations and reads $ND / 8$ bytes. Rescoring performs $O(k \cdot \texttt{oversample} \cdot D)$ operations and reads $k \cdot \texttt{oversample} \cdot D$ int8 values. By comparison, an exact float32 scan performs $O(ND)$ float operations over $4ND$ bytes.

## In the wild

- The Hugging Face blog post on Sentence Transformers ("Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval") describes this exact pipeline. It thresholds normalized embeddings at 0, packs them with `np.packbits`, retrieves `rescore_multiplier * top_k` candidates, and rescores them with the float32 query against int8 embeddings loaded from disk. Using a multiplier of 4, the authors report about 96% of float32 retrieval quality with rescoring, compared to about 92.5% without it.
- In Sentence Transformers, `quantize_embeddings(..., precision="ubinary")` calls `np.packbits(embeddings > 0)`, and `precision="binary"` shifts those same bytes to int8. Both `semantic_search_faiss` and `semantic_search_usearch` enable rescoring by default with `rescore_multiplier=2`.
- On the seeded test corpus in this problem, the Hamming stage alone recovers 61% of the true top 10, while 4x oversampling with int8 rescoring recovers 96%.
