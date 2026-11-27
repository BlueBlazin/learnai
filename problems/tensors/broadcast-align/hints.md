Before writing any arithmetic, write the two operand shapes one above the other, **aligned at the right edge**, and read down each column. For `scale_rows`, compare `(B, T, D)` above `(B,)`. Which axis of `x` does `s` sit under, and what shape does `s` need so its `B` axis sits under `x`'s `B` axis?

---

Reshape each operand so every axis meets either its matching partner or a size-1 axis.

- `relative_positions`: `i` varies down the rows and `j` varies across the columns, so form a column `(T, 1)` and a row `(1, T)`.
- `scale_rows`: reshape `s` to `(B, 1, 1)`.
- `pairwise_sq_dist`: the output keeps `N` and `M` and sums out `D`, so reshape `a` and `b` to broadcast to `(N, M, D)`.

Indexing with `None` inserts a size-1 axis at that position.

---

The key lines and their operand shapes:

```
pos = arange(T)
pos[:, None] - pos[None, :]            # (T, 1) - (1, T)       -> (T, T)

x * s[:, None, None]                   # (B, T, D) * (B, 1, 1) -> (B, T, D)

diff = a[:, None, :] - b[None, :, :]   # (N, 1, D) - (1, M, D) -> (N, M, D)
(diff * diff) summed over the last axis                        -> (N, M)
```
