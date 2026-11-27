For each function, write out the input's elements in row-major order (the order you read them in), and then write out the desired output's elements in row-major order. If the two lists match, a single reshape is all you need. If they differ, no reshape alone can produce the answer.

---

- `group_columns`: Reading the output in row-major order gives `w[0, 0], w[0, 1], ...`, which is the same order as the input. One reshape is enough.
- `interleave`: Reading the output in order gives `a0, b0, a1, b1, ...`, which no reshape of `a` followed by `b` can produce. Instead, first stack `a` and `b` along a new **last** axis of size 2 so `a[..., i]` and `b[..., i]` sit side by side, then merge the last two axes with a reshape.
- `patchify`: Split `H` into `(H / p, p)` and `W` into `(W / p, p)` with one reshape, giving five axes `(b, i, u, j, v)`. A patch index is `(i, j)` and an in-patch pixel is `(u, v)`. Remember that `reshape` can only merge neighboring axes.

---

The key lines:

```
w.reshape(R, C // g, g)

stack((a, b), on a new last axis)        # (..., n, 2)
    .reshape(..., 2 * n)

x.reshape(B, H // p, p, W // p, p)       # axes (b, i, u, j, v)
    .permute to (b, i, j, u, v)          # permute(0, 1, 3, 2, 4), or transpose(0, 1, 3, 2, 4) in JAX and MLX
    .reshape(B, (H // p) * (W // p), p * p)
```
