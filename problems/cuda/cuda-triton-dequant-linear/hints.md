For a given column `c` of output row `o`, derive three integer expressions in `c` alone: which `uint32` word holds its code, which 4-bit position inside that word holds the code, and which group holds its scale and bias. Notice which of these three change partway through a block of 16 columns when `group_size` is 24. Also consider what state the kernel needs to keep once column `c` has been added to the running dot product.

---

Reuse the blocked loop from `cuda-triton-rmsnorm` with `cols = start + tl.arange(0, BLOCK)`, `mask = cols < n_in`, and a `BLOCK`-element `float32` accumulator. Advance each row pointer by its row index times its row length: `batch_row * n_in` for `x`, `out_row * (n_in // 8)` for `words`, and `out_row * (n_in // group_size)` for `scales` and `biases`. Inside the loop, issue four masked loads with `other=0` or `other=0.0`: activations at `x_row + cols`, packed words at `words_row + cols // 8`, and scales and biases at their row pointers plus `cols // group_size`. After the loop, store `tl.sum(acc, axis=0)` to `y_ptr + batch_row * n_out + out_row`.

---

```python
word = tl.load(words_row + cols // 8, mask=mask, other=0)  # uint32
code = (word >> ((cols % 8) * 4).to(tl.uint32)) & 0xF
group = cols // group_size
scale = tl.load(scales_row + group, mask=mask, other=0.0).to(tl.float32)
bias = tl.load(biases_row + group, mask=mask, other=0.0).to(tl.float32)
acc += x * (code.to(tl.float32) * scale + bias)
```

Keep `word` unsigned so `>>` shifts in zeros rather than copying a sign bit (`tritonshim` rejects right shifts on negative signed values). Keep `bias` inside the parentheses because it is part of the dequantized weight, and load `x` with `other=0.0` so masked-off columns contribute zero.
