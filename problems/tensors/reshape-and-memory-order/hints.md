For each function, write the input's elements in row-major order (the order you would
read them in), then write the elements of the wanted output in row-major order. If the
two lists are the same, a reshape is all you need. If they differ, a reshape cannot be
the whole answer, whatever shape you give it.

---

- `group_columns`: the output read in order is `w[0, 0], w[0, 1], ...`: the same list.
  One reshape.
- `interleave`: the output read in order is `a0, b0, a1, b1, ...`. No reshape of `a`
  followed by `b` gives that. But if you first build a tensor in which `a[..., i]` and
  `b[..., i]` are neighbors along a new **last** axis of size 2, the list is already in
  the right order, and merging the last two axes finishes it.
- `patchify`: split `H` into `(H / p, p)` and `W` into `(W / p, p)` with one reshape.
  You now have five axes `(b, i, u, j, v)`. A patch is `(u, v)` and a patch number is
  `(i, j)`, and reshape can only merge axes that are neighbors.

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
