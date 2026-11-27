Two questions settle the design. Given `length`, where do `T` new tokens go and what is `length` afterwards? When `cache_attend` runs, which slots hold valid tokens and which do not? Work out the exact slot indices for a buffer of capacity 8 with `length = 5` and `T = 2` before writing code.

---

In `cache_write`, the new entries go into slots `length` through `length + T - 1` along axis 2, and the new length is `length + T` (not `length + 1`). Check `length + T > capacity` first and raise `ValueError`. Write with slice assignment in PyTorch and MLX (`k[:, :, length : length + T] = k_new`) or `k.at[:, :, length : length + T].set(k_new)` in JAX so the buffer keeps its shape.

In `cache_attend`, the valid tokens sit in slots `0` through `length - 1`. The simplest way to ignore the unused tail is to never read it. Slice both buffers to `[:, :, :length]` before doing any arithmetic. Slicing returns a view, so no data is copied.

---

After slicing, this is the attention step from `kv-cache-append` with `S = length - T` earlier tokens. Query `i` sits at position `length - T + i`, so:

```
visible[i, j] = (j <= length - T + i)      # shape (T, length)
out = ops.attention(q, k[:, :, :length], v[:, :, :length], visible)
```

If you keep the buffers full-sized and mask instead, masking the scores alone is not enough. The unused tail of `v` still enters the weighted sum, and a weight of 0 times NaN or infinity produces NaN. You must also replace the unused tail of `v` (and use `where` on the scores, since `NaN + (-inf)` is still `NaN`).
