Two codes differ in a bit exactly where their XOR has a 1. What remains is counting
the 1 bits of every byte of an `(N, D // 8)` tensor. A byte has only 256 possible
values: how could you count the bits of all of them at once without any bit
arithmetic in the loop?

---

`binarize`: compare with zero, reshape the last axis to `(D // 8, 8)`, multiply by
the bit values `[128, 64, 32, 16, 8, 4, 2, 1]`, sum, and cast to uint8.

`hamming_topk`: build a table of 256 integers, `table[b]` = number of set bits of
`b`. Then `table[xor(corpus_bits, q_bits)]` has one count per byte; sum over the
last axis for the distances. A stable ascending sort of the distances gives the
tie rule for free; keep the first `m`.

---

`search` is three lines once the other two exist:

```python
candidates, _ = hamming_topk(binarize(q), corpus_bits, k * oversample)
scores = distance(q, corpus_int8[candidates])  # only these rows are scored
order = stable_argsort_descending(scores)[:k]
return candidates[order], scores[order]
```

Index the int8 matrix with the candidate indices *before* calling `distance`: what
you pass in is what is counted. Sort the scores in descending order with a stable
sort (in MLX and JAX, sort the negated scores), so that equal scores keep the
candidate order.
