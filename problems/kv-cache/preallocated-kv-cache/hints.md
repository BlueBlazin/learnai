Two questions settle the design. Where do `T` new tokens go, given `length`, and what
is the length afterwards? And when `cache_attend` runs, which slots hold tokens and
which do not? Answer both with slot indices before writing any code, for a buffer of
capacity 8 with `length = 5` and `T = 2`.

---

`cache_write`: the new entries go to slots `length` to `length + T - 1`, one slice
along axis 2, and the new length is `length + T` (not `length + 1`). Check
`length + T > capacity` first and raise `ValueError`. The write itself is a slice
assignment in PyTorch and MLX (`k[:, :, length:end] = k_new`) and
`k.at[:, :, length:end].set(k_new)` in JAX; either way the buffer keeps its shape.

`cache_attend`: the tokens are in slots `0` to `length - 1`. The simplest way to make
the tail irrelevant is to never read it: slice both buffers to `[:, :, :length]`
before doing any arithmetic. A slice is a view, so nothing is copied.

---

After slicing, the problem is the attention step of `kv-cache-append` with
`S = length - T` earlier tokens: query `i` sits at position `length - T + i`, so

```
visible[i, j] = (j <= length - T + i)      # shape (T, length)
out = ops.attention(q, k[:, :, :length], v[:, :, :length], visible)
```

If you prefer to keep the buffers whole and mask instead, masking the scores is not
enough. The tail of `v` still enters the weighted sum, and a weight of exactly 0
times a NaN or an infinity is NaN. You would have to replace the tail of `v` as well
(and use `where` on the scores, since NaN plus `-inf` is NaN).
