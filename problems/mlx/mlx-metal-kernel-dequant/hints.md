Give every weight its own thread. A thread knows only its position in the grid, so
make the position *be* the weight's coordinates: which grid shape makes
`thread_position_in_grid.x` the column and `.y` the row? Then everything a thread
needs is integer arithmetic on those two numbers: which word, which bits of it, which
group.

---

With `p = 32 / bits` codes per word, the thread for `(row, col)` reads word
`col / p` of its row, shifts it right by `bits * (col % p)`, and masks with
`(1 << bits) - 1`. Its group is `col / group_size`. All three buffers are row-major,
so the flat indices are `row * words_per_row + col / p` and
`row * groups_per_row + col / group_size`, and the output index is
`row * n_in + col`. Keep the word and the code `uint`: a right shift of a signed
integer copies the sign bit. Pass `bits` as a template int, read `words_shape[1]`
and `scales_shape[1]` for the row widths, and compute in `float`, converting to the
output type `T` at the end.

---

The body:

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

launched with `template=[("T", scales.dtype), ("BITS", bits)]`,
`grid=(n_in, n_out, 1)`, a threadgroup such as `(32, 8, 1)`,
`output_shapes=[(n_out, n_in)]`, and `output_dtypes=[scales.dtype]`. Leave
`ensure_row_contiguous` at its default so strided inputs are copied first.
