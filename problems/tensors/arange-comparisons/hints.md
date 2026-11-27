Each mask is a table of true and false indexed by two things. Name them for each
function (a row and a position, a row index and a column index, two positions) and
write the condition for one entry as a comparison of two integers. Then make each of
the two integers a vector, and give the vectors different axes so that the comparison
broadcasts to the table.

---

- `length_mask`: the table is `(B, T)`. The position `j` varies along axis 1, so it
  is `arange(T)` as a `(1, T)` row; the length varies along axis 0, so it is a `(B, 1)`
  column. For left padding, the first real position of a row is `T - lengths[b]`.
- `band_mask`: with `i` a `(T, 1)` column and `j` a `(1, T)` row, `i - j` is the whole
  `(T, T)` table of "how far back". The condition has two parts, joined with `&`,
  and each part needs its own parentheses.
- `same_segment`: the two vectors are the same vector, once as a column and once as a
  row.

---

The key lines, in PyTorch (JAX and MLX are the same with `jnp.arange` and
`mx.arange`):

```
pos = torch.arange(T)[None, :]          # (1, T)
n = lengths[:, None]                    # (B, 1)
pos < n                                 # side == "right"
pos >= T - n                            # side == "left"

back = torch.arange(T)[:, None] - torch.arange(T)[None, :]
(back >= 0) & (back < w)                # band_mask

seg[:, None] == seg[None, :]            # same_segment
```
