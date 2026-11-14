First find the position, then worry about the indexing. For each padding side, write
down where the real tokens of a row with `lengths[b] = 2` and `T = 5` sit, and the
index of the last one. One of the two answers does not depend on the length at all.

---

You now have one position per row: an integer tensor `last` of shape `(B,)`. You need
`logits[b, last[b]]` for every `b`, which pairs each row with **its own** position.
`logits[:, last]` does not do that: a slice on the first axis combined with an index on
the second selects every listed position for every row, shape `(B, B, V)`.

---

Index both axes with tensors of the same shape, so that they are paired elementwise:

```
rows = arange(B)
out  = logits[rows, last]        # (B, V): out[b] = logits[rows[b], last[b]]
```

with `last = lengths - 1` for right padding and `last = T - 1` for left padding. An
equivalent form is `take_along_axis` / `gather` along axis 1 with an index of shape
`(B, 1, V)` (or `(B, 1, 1)` where broadcasting is supported), followed by dropping
axis 1. Avoid selecting by multiplying with a one-hot mask and summing: `0 * -inf` is
NaN.
