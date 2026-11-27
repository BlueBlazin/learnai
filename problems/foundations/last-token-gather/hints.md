Find the target position for each row first, then solve the indexing. For both padding sides, write down where the real tokens sit when `lengths[b] = 2` and `T = 5`, and note the index of the last real token. One of the two cases does not depend on `lengths[b]` at all.

---

Once you have an integer tensor `last` of shape `(B,)` holding one target position per row, you need `logits[b, last[b]]` for every `b`. Writing `logits[:, last]` does not pair rows with their own positions. Combining a slice on axis 0 with a 1D index tensor on axis 1 selects every listed position for every row, producing shape `(B, B, V)`.

---

Index both axes with 1D tensors of the same shape `(B,)` so they pair elementwise:

```
rows = arange(B)
out  = logits[rows, last]        # (B, V): out[b] = logits[rows[b], last[b]]
```

Set `last = lengths - 1` for right padding and `last = T - 1` for left padding. An equivalent approach uses `take_along_axis` or `gather` along axis 1 with an index of shape `(B, 1, V)` (or `(B, 1, 1)` where broadcasting is supported), then drops axis 1. Do not select by multiplying with a one-hot mask and summing, because `0 * -inf` produces `NaN`.
