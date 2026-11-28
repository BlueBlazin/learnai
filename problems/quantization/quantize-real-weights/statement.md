Implement MLX-style group-wise affine quantization for every weight matrix in IBM's Granite 4.0 350M model, and measure its quantization cost. Write `quantize_dequantize` to quantize and dequantize one matrix, and `quantize_report` to compute per-layer error, pooled model error, effective bits per weight, and the three worst layers.

## Specification

```python
def quantize_dequantize(w, bits, group_size): ...
def quantize_report(weights, bits, group_size): ...
```

### `quantize_dequantize(w, bits, group_size)`

`w` is a float32 weight matrix of shape `(out, in)`. Split each row into `in / group_size` consecutive groups of `group_size` inputs and quantize each group independently. Let $L = 2^{\text{bits}} - 1$ be the largest unsigned integer code, and let $m$ and $M$ be the minimum and maximum values of a group. For each group, compute the following steps:

1. **Initial step size.** Set $s_0 = \max\!\big((M - m) / L,\ 10^{-7}\big)$. The floor at $10^{-7}$ keeps constant groups finite so they decode back to their original value with all-zero codes.
2. **Anchor edge.** Start the quantization grid at the endpoint further from zero. If $|m| > |M|$, set the anchor $e = m$ and scale $s = s_0$ so codes count upward from the minimum. Otherwise, set $e = M$ and $s = -s_0$ so codes count downward from the maximum.
3. **Zero on the grid.** Compute $q_0 = \operatorname{round}(e / s)$. If $q_0 \ne 0$, replace the scale with $s = e / q_0$ and set the bias to $b = e$. If $q_0 = 0$, keep $s$ unchanged and set $b = 0$.
4. **Integer codes.** For each weight $w_j$ in the group, compute $c_j = \operatorname{clamp}\big(\operatorname{round}((w_j - b) / s),\ 0,\ L\big)$.
5. **Dequantized weights.** Reconstruct each weight as $\hat{w}_j = c_j \cdot s + b$.

Return the dequantized matrix as a float32 tensor with the same shape as `w`. Here $\operatorname{round}$ rounds to the nearest integer with ties to even, matching `round` in PyTorch, JAX, and MLX.

Step 3 is what distinguishes MLX's quantizer from the textbook affine scheme. When $q_0 \ne 0$, integer code $-q_0$ decodes to $-q_0 \cdot (e / q_0) + e = 0$, so zero is represented exactly. Adjusting $s$ can shrink the step slightly, which may push values near the opposite end of the range past code $L$. The clamp in step 4 clips those values back into $[0, L]$.

### `quantize_report(weights, bits, group_size)`

`weights` maps tensor names to float32 tensors. It may be a plain `dict` or the kit's lazy [`Weights`](../../../python/src/learnai/kit/realmodel.py) mapping, which is read-only and supports iteration and `.items()`. Quantize every 2-D tensor (linear layers, the embedding matrix, and a separate output head if present). Leave all other tensors, such as 1-D norm gains, unchanged.

Return a `dict` with four keys whose numeric values are Python `float`s:

- `"sqnr_db"`: a `dict` mapping each 2-D tensor name to its signal-to-quantization-noise ratio in decibels,
  $$
  10 \log_{10} \frac{\sum_k w_k^2}{\sum_k (w_k - \hat{w}_k)^2},
  $$
  where $w_k$ and $\hat{w}_k$ are the original and dequantized elements of that matrix.
- `"model_sqnr_db"`: the same ratio pooled across all quantized matrices. Sum the signal powers of all matrices for the numerator and sum their noise powers for the denominator. Do not average per-layer decibel values, since pooling weights each matrix by its total signal power.
- `"bits_per_weight"`: total stored bits of the quantized model divided by its total parameter count, counting **all** tensors in `weights`. A quantized matrix of $n$ elements takes $n \cdot \text{bits}$ bits for its codes plus one 16-bit scale and one 16-bit bias per group ($32 \cdot n / \texttt{group\_size}$ additional bits). An unquantized tensor takes 16 bits per element. Every element of every tensor counts as one parameter.
- `"worst"`: a list of the three matrix names with the lowest SQNR, ordered from lowest to highest.

Implement the quantizer yourself. Calling `mx.quantize`, `mx.dequantize`, `nn.quantize`, or MLX's quantized layers (`QuantizedLinear`, `QuantizedEmbedding`) is not allowed.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)`, `in` divisible by `group_size` | float32 |
| return value of `quantize_dequantize` | `(out, in)` | float32 |
| values of `weights` | `(out, in)` for matrices, `(d,)` for gains | float32 |

## Example

With 2 bits ($L = 3$) and `group_size = 4` on a single row:

```python
w = [[-1.0, -0.4, 0.2, 0.5, -0.7, -0.1, 0.3, 0.6]]
quantize_dequantize(w, bits=2, group_size=4)
# [[-1.0, -0.5, 0.0, 0.5, -0.7, 0.0, 0.35, 0.35]]
```

In the first group, $m = -1.0$ and $M = 0.5$, so $s_0 = (0.5 - (-1.0)) / 3 = 0.5$. Because $|-1.0| > |0.5|$, the anchor is $e = -1.0$ with $s = 0.5$. Then $q_0 = \operatorname{round}(-1.0 / 0.5) = -2$, so $s = -1.0 / -2 = 0.5$ (unchanged) and $b = -1.0$. The codes are $0, 1, 2, 3$ (for example, $-0.4$ maps to $\operatorname{round}(0.6 / 0.5) = 1$), producing the grid $-1.0, -0.5, 0.0, 0.5$, which already contains zero.

In the second group, $m = -0.7$ and $M = 0.6$, so $s_0 = 1.3 / 3 = 0.4333$ and $e = -0.7$. Computing $q_0 = \operatorname{round}(-0.7 / 0.4333) = \operatorname{round}(-1.615) = -2$ updates the scale to $s = -0.7 / -2 = 0.35$. The grid becomes $-0.7, -0.35, 0.0, 0.35$. The weight $0.6$ would require code $\operatorname{round}(1.3 / 0.35) = 4$, which clamps to $3$ and decodes as $0.35$. A textbook grid of $-0.7, -0.267, 0.167, 0.6$ would have preserved $0.6$ at the cost of losing zero.

Across this matrix, the signal power is $2.40$ and the noise power is $0.125$, giving an SQNR of $10 \log_{10}(19.2) = 12.83$ dB.

The fourth public test case runs `quantize_report` on a small random model with Granite's tensor names (14 matrices totaling 221,184 elements and 5 gain vectors totaling 640 elements). At 4 bits with `group_size = 64`, the quantized model occupies $221184 \cdot (4 + 32/64) + 640 \cdot 16 = 1005568$ bits, which is $4.5332$ bits per weight rather than $4.0$.

## Constraints

- `quantize_dequantize` is compared against MLX 0.32.4 (`mx.quantize` followed by `mx.dequantize` on CPU). At most 2% of elements in a matrix may differ from MLX, and none by more than about one step of its group. The worked example is checked to `1e-5`. This allowance accounts for real weights that fall exactly halfway between two codes, where a 1-bit difference in the quotient flips the rounding direction without changing error magnitude.
- SQNR values are checked to a relative tolerance of `1e-3` (and `1e-3` dB absolute), `bits_per_weight` to `1e-6`, and `"worst"` for an exact match. Compute sums of squares accurately for matrices with up to $10^8$ elements, and count bits in Python integers since the real model occupies roughly $3 \cdot 10^9$ bits.
- $2 \le \text{bits} \le 8$, and `group_size` divides `in`. Tests use group sizes 32, 64, and 128 (the sizes MLX supports), plus 4 in the worked example. Groups may be constant, all zero, or entirely on one side of zero. `weights` contains at least three matrices, all with distinct SQNR values and nonzero quantization error.
- The real model has 169 matrices (352.3 M weights) and 57 gain vectors, tested at `group_size = 64` for both 8 bits and 4 bits. The kit runs the quantized model on a 12-token prompt and verifies that the mean next-token KL divergence from the original model is at most **0.01 nats at 8 bits** and **2.5 nats at 4 bits**.
- The time limit is 30 seconds per case. One pass over the real model takes about a second.

> [!WARNING]
> In JAX, dividing an array by a Python scalar multiplies by the scalar's reciprocal, which differs by one bit for about a third of all float32 values. In step 1, that single bit can flip a tie in $q_0$ and shift an entire group onto a different grid. Divide by an array (`jnp.full_like(x, L)`) wherever the quotient feeds `round`.

## Why this matters

"4-bit" is a label, not the actual memory footprint. When `mlx_lm.convert -q` quantizes this model at 4 bits with groups of 64, each group adds a 16-bit scale and a 16-bit bias, bringing the true cost to 4.5 bits per weight. Treating the model as 4.0 bits per weight underestimates memory usage by 12%.

Error reporting is just as easy to distort. Averaging decibel values across layers gives a 16,000-weight projection the same influence as a 100-million-weight embedding, and omitting the embedding skips 29% of the model. Per-layer SQNR tables are also how quantization failures get diagnosed in practice. On Granite 350M, 8-bit quantization shifts next-token predictions by only 0.003 nats, whereas 4-bit shifts them by 1.5 nats, breaking the model even though every layer still reports around 20 dB. Choosing which layers to keep at higher precision starts with the exact report this function builds.
