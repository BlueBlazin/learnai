Assign one GPU thread to each output weight. Because a thread only knows its grid coordinates, choose a 2D grid shape where `thread_position_in_grid.x` is the column and `thread_position_in_grid.y` is the row. From those two coordinates, use integer arithmetic to find the packed word, the bit offset inside that word, and the scale/bias group.

---

With `p = 32 / bits` codes per word and `n_in` weights per row, the thread for `(row, col)` reads word `col / p` in its row, shifts right by `bits * (col % p)`, and masks with `(1 << bits) - 1`. Its group index within the row is `col / group_size`. Since all buffers are row-major, the flat indices are `row * words_per_row + col / p` for `words`, `row * groups_per_row + col / group_size` for `scales` and `biases`, and `row * n_in + col` for the output. Keep `word` and `code` as `uint` so right shifts fill with zeros instead of copying the sign bit. Pass `bits` as a template int, read `words_shape[1]` and `scales_shape[1]` for the row widths, compute in `float`, and cast to `T` once when writing the output.

---

The kernel body:

```cpp
uint col = thread_position_in_grid.x;
uint row = thread_position_in_grid.y;
uint per_word = 32 / BITS;
uint n_words = words_shape[1];
uint n_groups = scales_shape[1];
uint n_in = n_words * per_word;

uint word = words[row * n_words + col / per_word];
uint code = (word >> (BITS * (col % per_word))) & ((1u << BITS) - 1u);
uint group = row * n_groups + col / (n_in / n_groups);
out[row * n_in + col] = T(fma(float(code), float(scales[group]), float(biases[group])));
```

Launch it with `template=[("T", scales.dtype), ("BITS", bits)]`, `grid=(n_in, n_out, 1)`, a threadgroup such as `(32, 8, 1)`, `output_shapes=[(n_out, n_in)]`, and `output_dtypes=[scales.dtype]`. Keep `ensure_row_contiguous` at its default so strided views are copied before the kernel runs.
