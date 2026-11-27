Program `r` owns elements `r * n_cols` through `r * n_cols + n_cols - 1` of `x` and `out`. A block of `BLOCK` columns starting at `start` covers columns `start` through `start + BLOCK - 1`. In the last block of a row, which of those columns are valid, and what sits in memory at the invalid offsets for row 0 versus the last row? Express the validity condition as a comparison between two values the kernel already has.

---

Use two loops over `range(0, n_cols, BLOCK)`. In both loops, compute `cols = start + tl.arange(0, BLOCK)` and `mask = cols < n_cols`. In the first loop, load `x` with `mask` and `other=0.0`, cast to `tl.float32`, and accumulate `x * x` into a `BLOCK`-element `float32` vector. After the loop, compute `tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)`. In the second loop, load `x` and `w` with the same mask, cast to `tl.float32`, and store `x / rms * w` with `mask=mask`. In `rms_norm`, reshape `x` to `(-1, n_cols)` and call `.contiguous()` on both `x` and `weight`, allocate `out` with `torch.empty_like`, launch on grid `(rows,)`, and reshape the output back to `x.shape`.

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

Call `.to(tl.float32)` before squaring, because multiplying two `float16` values produces a `float16` that can overflow. The second pass uses the same loop structure and ends with `tl.store(out_row + cols, x / rms * w, mask=mask)`. Remember that `w` is shared across all rows, so load it from `w_ptr + cols` with no row offset.
