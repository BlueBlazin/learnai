After a batch passes through the model, which column's logits do you read to sample the next token for each row? What must sit in that column for every row, including the shortest one?

---

Validate `side`, check that `seqs` is non-empty, compute `lengths = [len(seq) for seq in seqs]`, and verify that no length is zero. Then let `T = max(lengths)`. For each sequence of length `n`, build `T - n` copies of `pad_id` for the ids, `T - n` zeros for the mask, and `n` ones for the mask, then concatenate them in the order requested by `side`.

---

For left padding, each row is built as:

```python
ids_row = [pad_id] * (T - n) + list(seq)
mask_row = [0] * (T - n) + [1] * n
```

For right padding, reverse the order of the two halves. Using `list(seq)` guarantees a fresh copy even when a row needs no padding. Always build `mask_row` from `n` rather than from `ids_row` so that real tokens equal to `pad_id` keep mask `1`.
