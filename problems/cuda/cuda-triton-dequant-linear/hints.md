For one column `c` of output row `o`, three things must be found from `c` alone:
which word holds its code, where in that word the code sits, and which scale and
bias apply. Write each as an integer expression in `c`. Which of them change in the
middle of a block of 16 columns when `group_size` is 24? And what does the kernel
have to keep in memory once column `c` has been added to the sum?

---

Use the loop of `cuda-triton-rmsnorm`: `cols = start + tl.arange(0, BLOCK)`,
`mask = cols < n_in`, a float32 accumulator of `BLOCK` partial sums. Per block, four
masked loads, all with `other=` zero: the activations at `x_row + cols`; the words at
`words_row + cols // 8` (each word is loaded eight times, once per column that needs
it); the scales and the biases at `row pointer + cols // group_size`. The row
pointers are the tensor pointers advanced by the program's row times the row length:
`batch_row * n_in` for `x`, `out_row * (n_in // 8)` for the words,
`out_row * (n_in // group_size)` for scales and biases. After the loop, one store of
`tl.sum(acc, axis=0)` at `y_ptr + batch_row * n_out + out_row`.

---

```python
word = tl.load(words_row + cols // 8, mask=mask, other=0)  # uint32
code = (word >> ((cols % 8) * 4).to(tl.uint32)) & 0xF
group = cols // group_size
scale = tl.load(scales_row + group, mask=mask, other=0.0).to(tl.float32)
bias = tl.load(biases_row + group, mask=mask, other=0.0).to(tl.float32)
acc += x * (code.to(tl.float32) * scale + bias)
```

Keep the word unsigned: on a signed integer `>>` may copy the sign bit (C does; Triton's
documentation does not say, so the shim refuses a negative value there). The bias is
inside the parentheses (it is part of the weight), and `x` was loaded with
`other=0.0`, so the masked columns contribute exactly nothing.
