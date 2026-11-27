Before writing any arithmetic, write the two shapes one under the other, **aligned at
the right edge**, and read the columns. For `scale_rows` that is `(B, T, D)` over
`(B,)`: which axis of `x` does the only axis of `s` sit under? Which shape would `s`
need so that its `B` sits under the `B` of `x`?

---

Each function needs its operands reshaped so that every axis meets either an axis of
the same meaning or a 1.

- `relative_positions`: `i` varies down the rows and `j` along the columns, so you need
  the positions once as a column `(T, 1)` and once as a row `(1, T)`.
- `scale_rows`: `s` must become `(B, 1, 1)`.
- `pairwise_sq_dist`: the result has an `N` axis and an `M` axis, and `D` is summed
  away. Give `a` and `b` shapes that broadcast to `(N, M, D)`.

Indexing with `None` adds an axis of size 1 at that place.

---

The key lines, with the shape of each operand:

```
pos = arange(T)
pos[:, None] - pos[None, :]            # (T, 1) - (1, T)       -> (T, T)

x * s[:, None, None]                   # (B, T, D) * (B, 1, 1) -> (B, T, D)

diff = a[:, None, :] - b[None, :, :]   # (N, 1, D) - (1, M, D) -> (N, M, D)
(diff * diff) summed over the last axis                        -> (N, M)
```
