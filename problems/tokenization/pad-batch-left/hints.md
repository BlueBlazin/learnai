After the batch goes through the model, which position's logits do you read to pick the
next token of each row? What has to be stored at that position for every row, including
the shortest one?

---

Compute the lengths first, then the width `T = max(lengths)`. For each sequence build
two short lists: `T - n` copies of `pad_id` for the ids and `T - n` zeros for the mask,
plus `n` ones. Concatenate them in the order that `side` asks for. Do the validation
before any of this: `side` must be one of the two strings, the batch must not be empty,
and no length may be zero.

---

For the left side the two rows are

```python
ids_row = [pad_id] * (T - n) + list(seq)
mask_row = [0] * (T - n) + [1] * n
```

and for the right side the same pieces in the other order. `list(seq)` makes a copy, so
a row that needs no padding is still a new list. Build the mask from `n`, not from
`ids_row`: a token equal to `pad_id` inside the sequence must stay visible.
