Each mask is a 2D table of booleans indexed by two quantities: a row and a position, a row index and a column index, or two positions. Write the condition for one cell as a comparison of two integers, then turn each integer into a 1D vector on a different axis so the comparison broadcasts across the table.

---

- `length_mask`: the output has shape `(B, T)`. Position `j` varies along axis 1, so put `arange(T)` in a `(1, T)` row. Length varies along axis 0, so put `lengths` in a `(B, 1)` column. For left padding, the first real token in row `b` is at `T - lengths[b]`.
- `band_mask`: with `i` as a `(T, 1)` column and `j` as a `(1, T)` row, `i - j` gives the `(T, T)` table of distances behind `i`. Join the two bounds with `&`, wrapping each comparison in parentheses.
- `same_segment`: broadcast the same `seg` vector as a `(T, 1)` column and a `(1, T)` row.

---

The key lines in PyTorch (JAX and MLX are identical with `jnp.arange` and `mx.arange`):

```
pos = torch.arange(T)[None, :]          # (1, T)
n = lengths[:, None]                    # (B, 1)
pos < n                                 # side == "right"
pos >= T - n                            # side == "left"

back = torch.arange(T)[:, None] - torch.arange(T)[None, :]
(back >= 0) & (back < w)                # band_mask

seg[:, None] == seg[None, :]            # same_segment
```
