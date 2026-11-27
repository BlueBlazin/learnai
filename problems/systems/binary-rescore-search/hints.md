Two binary codes differ at a bit position wherever their bitwise XOR has a `1`. After XORing, you need to count the `1` bits in every byte of an `(N, D // 8)` tensor. Since a byte has only 256 possible values, how can you look up all byte bit counts at once without bit-twiddling loops?

---

For `binarize`, compare `e` with zero, reshape the last axis to `(D // 8, 8)`, multiply by `[128, 64, 32, 16, 8, 4, 2, 1]`, sum over the last axis, and cast to uint8.

For `hamming_topk`, build a 256-entry lookup table where `table[b]` is the number of set bits in `b`. Indexing `table[xor(corpus_bits, q_bits)]` gives one bit count per byte, and summing over the last axis gives the Hamming distances. A stable ascending sort of those distances handles the tie-breaking rule automatically, and slicing the first `m` entries selects the candidates.

---

Once `binarize` and `hamming_topk` work, `search` is four lines:

```python
candidates, _ = hamming_topk(binarize(q), corpus_bits, k * oversample)
scores = distance(q, corpus_int8[candidates])  # only these rows are scored
order = stable_argsort_descending(scores)[:k]
return candidates[order], scores[order]
```

Index `corpus_int8` with `candidates` *before* calling `distance`, because every row passed to `distance` is counted. Sort the scores in descending order with a stable sort (in MLX and JAX, sort the negated scores) so that equal scores preserve the candidate order.
