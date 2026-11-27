Think about the codebook and the quantizer separately. Pushing evenly spaced probabilities through the inverse normal CDF yields levels that are unevenly spaced in value. If you use the same number of probabilities on both sides, does any level land on exact zero? Once a block is divided by its largest magnitude, every value lies in $[-1, 1]$. Since the codes are table indices rather than positions on a uniform grid, how can you find the nearest entry of a 16-element table for all weights at once?

---

To build the table, take 8 probabilities for the positive half and 7 for the negative half using `linspace(0.9677083, 0.5, 9)[:-1]` and `linspace(0.9677083, 0.5, 8)[:-1]`. Dropping the last element removes $0.5$, whose quantile is $0$. Compute the inverse normal CDF with `torch.special.ndtri` or `jax.scipy.special.ndtri` (in MLX, use $\Phi^{-1}(p) = \sqrt{2}\,\operatorname{erfinv}(2p - 1)$). Concatenate the negated negative half, a single $0.0$, and the reversed positive half so the 16 values are ascending, then divide by the maximum.

For `nf4_quantize`, reshape `w` to `(out, in // block_size, block_size)`, take the maximum absolute value along the last axis, and replace any zero maximum with `1` before dividing.

---

Add a trailing dimension to the normalized blocks and broadcast-subtract the 16-entry table to get shape `(out, in // block_size, block_size, 16)`. The nearest codebook index is:

$$
\text{codes} = \operatorname*{arg\,min}_{k}\ \big|\, \text{normalized}[\dots, \text{None}] - \text{levels}\, \big|
$$

taken over the last axis. Cast `codes` to uint8 and reshape to `(out, in)`.

For `nf4_dequantize`, index into `levels` with `codes` (cast to an integer index dtype) and multiply by `absmax` repeated `block_size` times consecutively along the input axis (`repeat_interleave`, not `tile`).
