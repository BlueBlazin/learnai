Separate the causal and padding rules and decide the shape of each before combining them. The causal rule compares a query index with a key index and is the same for every row in the batch. The padding rule depends on the batch row and a single position. Decide whether that position is the query or the key by remembering that a real token must never read from a padding position.

---

Build three boolean tensors from a single `arange(T)` and combine them with broadcasting:

- `real` of shape `(B, T)`, by comparing positions with `lengths[:, None]`. For right padding, the real positions are the first `lengths[b]`. For left padding, they are the last `lengths[b]`.
- `causal` of shape `(T, T)`, where entry `[i, j]` is true when `j <= i`.
- `diagonal` of shape `(T, T)`, where entry `[i, j]` is true when `j == i`.

Because `real` describes keys and keys are the **last** axis of the mask, reshape it to `real[:, None, :]` of shape `(B, 1, T)`. Check a left-padded example and verify that every query row has at least one allowed entry.

---

```
allowed = (causal[None, :, :] & real[:, None, :]) | diagonal      # (B, T, T), boolean
mask    = where(allowed, 0.0, -inf)                               # float32
result  = mask[:, None, :, :]                                     # (B, 1, T, T)
```

OR the diagonal in after the AND so that it survives for padding positions. Fill disallowed entries with true `-inf` using `where` or `masked_fill` instead of multiplying a boolean by `-inf`, because `0 * -inf` produces `NaN` in the allowed entries.
