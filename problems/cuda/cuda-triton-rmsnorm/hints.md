Program `r` owns elements `r * n_cols` to `r * n_cols + n_cols - 1` of `x` and `out`.
A block of `BLOCK` columns starting at column `start` covers columns `start` to
`start + BLOCK - 1`. For the last block of a row, which of those columns exist? What
is behind the pointer at the ones that do not, for row 0 and for the last row? Write
the condition "this column exists" as a comparison between two things the kernel
has.

---

Two loops over `range(0, n_cols, BLOCK)`. In both, `cols = start + tl.arange(0,
BLOCK)` and the mask is `cols < n_cols`. The first loop loads `x` with that mask and
`other=0.0`, converts to float32, and adds the squares to a float32 accumulator of
`BLOCK` elements; after it, `tl.sum(acc, axis=0) / n_cols + eps` goes under the
root. The second loop loads `x` and the gains again, and stores `x / rms * w` with
the same mask. For the wrapper: `x.reshape(-1, d).contiguous()`, an output from
`torch.empty_like`, one launch on a grid of `(rows,)`, and a `reshape` back.

---

```python
row = tl.program_id(0)
x_row = x_ptr + row * n_cols
acc = tl.zeros((BLOCK,), tl.float32)
for start in range(0, n_cols, BLOCK):
    cols = start + tl.arange(0, BLOCK)
    x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
    acc += x * x
rms = tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)
```

The `.to(tl.float32)` comes before the multiplication: the product of two float16
values is a float16. The second pass is the same loop with a `tl.store(out_row +
cols, x / rms * w, mask=mask)` at the end; the gains are at `w_ptr + cols`, with no
row offset.
