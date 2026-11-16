Quantize every matrix of a real model, IBM's Granite 4.0 350M, the way
`mlx_lm.convert -q` does, and report what it cost: the error of each layer, the error
of the model, the true number of bits per weight, and the layers that suffered most.
The quantizer is the group-wise affine scheme of `groupwise-4bit` with the two
refinements MLX adds to it, and the report is where the accounting mistakes live.

## Specification

```python
def quantize_dequantize(w, bits, group_size): ...  # the matrix after a round trip
def quantize_report(weights, bits, group_size): ...  # a dict
```

**`quantize_dequantize(w, bits, group_size)`**. `w` is a float32 matrix of shape
`(out, in)`. Each row is cut into `in / group_size` consecutive groups of `group_size`
inputs, and each group is quantized on its own. With $L = 2^{\text{bits}} - 1$, the
largest code, and $m$ and $M$ the minimum and maximum of the group:

1. **The step.** $s_0 = \max\!\big((M - m) / L,\ 10^{-7}\big)$.
2. **The edge.** The grid starts at the end of the range that is further from zero.
   If $|m| > |M|$: $e = m$ and $s = s_0$ (the codes count upward from the minimum).
   Otherwise: $e = M$ and $s = -s_0$ (the codes count downward from the maximum).
3. **Zero on the grid.** $q_0 = \operatorname{round}(e / s)$. If $q_0 \ne 0$, the
   scale is replaced by $s = e / q_0$ and the bias is $b = e$. If $q_0 = 0$, the scale
   stays and the bias is $b = 0$.
4. **Codes.** $c_j = \operatorname{clamp}\big(\operatorname{round}((w_j - b) / s),\ 0,\ L\big)$.
5. **The result.** $\hat w_j = c_j \cdot s + b$.

The function returns $\hat W$, a float32 tensor of the shape of `w`: the weights that
the stored codes, scales, and biases stand for. $\operatorname{round}$ rounds to the
nearest integer, ties to even, as `round` does in PyTorch, JAX, and MLX.

Step 3 is what the textbook scheme lacks. After it, the value 0 has the integer code
$-q_0$ and is reproduced exactly: $-q_0 \cdot (e / q_0) + e = 0$. The price is that the
step changed slightly, so the far end of the range may now need a code above $L$; the
clamp in step 4 cuts it off. The floor in step 1 keeps a group of equal values finite
(its codes are all 0 and it comes back exactly).

**`quantize_report(weights, bits, group_size)`**. `weights` maps tensor names to
float32 tensors (a dict, or the kit's lazy
[`Weights`](../../../python/src/learnai/kit/realmodel.py), a read-only mapping
with `.items()`). Every tensor with **two dimensions** is a matrix and is quantized: the
linear layers, and also the embedding matrix and a separate output head if there is
one. Every other tensor (the 1-D norm gains) is left as it is. The function returns a
dict with four keys, whose numbers are Python floats:

- `"sqnr_db"`: a dict from the name of each matrix to its signal-to-quantization-noise
  ratio in decibels,
  $$
  10 \log_{10} \frac{\sum_k w_k^2}{\sum_k (w_k - \hat w_k)^2},
  $$
  with both sums over all elements of that matrix.
- `"model_sqnr_db"`: the same ratio pooled over the model: the numerator is the sum of
  the numerators of all matrices, the denominator the sum of their denominators. This
  is not the mean of the per-layer decibel values; it weights each matrix by the
  power it holds.
- `"bits_per_weight"`: the bits the quantized model occupies, divided by its number
  of parameters. Both count **all** tensors in `weights`. A matrix of $n$ elements
  occupies $n \cdot \text{bits}$ for its codes plus one scale and one bias per group
  as 16-bit floats: $32 \cdot n / \texttt{group\_size}$ more. A tensor that is not
  quantized occupies 16 bits per element. A parameter is one element of any tensor.
- `"worst"`: the names of the three matrices with the lowest SQNR, lowest first.

Write the quantizer yourself: `mx.quantize`, `mx.dequantize`, `nn.quantize`, and
MLX's quantized layers are not allowed.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)`, `in` a multiple of `group_size` | float32 |
| result of `quantize_dequantize` | `(out, in)` | float32 |
| values of `weights` | `(out, in)` for matrices, `(d,)` for gains | float32 |

## Example

Two bits ($L = 3$) and groups of 4, on one row:

```python
w = [[-1.0, -0.4, 0.2, 0.5, -0.7, -0.1, 0.3, 0.6]]
quantize_dequantize(w, bits=2, group_size=4)
# [[-1.0, -0.5, 0.0, 0.5, -0.7, 0.0, 0.35, 0.35]]
```

First group: $m = -1$, $M = 0.5$, $s_0 = 0.5$. The minimum is further from zero, so
$e = -1$, $s = 0.5$. $q_0 = \operatorname{round}(-2) = -2$, so $s = -1 / -2 = 0.5$
(unchanged) and $b = -1$. The codes are $0, 1, 2, 3$ (for $-0.4$:
$\operatorname{round}(0.6 / 0.5) = 1$), and the grid $-1, -0.5, 0, 0.5$ contains zero.

Second group: $m = -0.7$, $M = 0.6$, $s_0 = 1.3 / 3 = 0.4333$, $e = -0.7$.
$q_0 = \operatorname{round}(-0.7 / 0.4333) = \operatorname{round}(-1.615) = -2$, so
the scale becomes $-0.7 / -2 = 0.35$: the grid is now $-0.7, -0.35, 0, 0.35$. The
value $0.6$ would need the code $\operatorname{round}(1.3 / 0.35) = 4$, is clamped
to 3, and comes back as $0.35$. The textbook grid $-0.7, -0.267, 0.167, 0.6$ would
have kept the maximum and lost the zero.

For this matrix the signal power is $2.40$ and the noise power $0.125$: an SQNR of
$10 \log_{10} 19.2 = 12.83$ dB.

The fourth public case reports on a small random model with Granite's tensor names:
14 matrices with 221,184 elements and 5 gain vectors with 640. At 4 bits and groups of
64 it occupies $221184 \cdot (4 + 32/64) + 640 \cdot 16 = 1005568$ bits, which is
4.5332 bits per weight, not 4.

## Constraints

- `quantize_dequantize` is compared with what MLX 0.32.4 computes (`mx.quantize`
  followed by `mx.dequantize`, on the CPU). At most 2 percent of the values of a
  matrix may differ from it, none by more than about one step of its group; the
  worked example is compared to `1e-5`. The allowance exists because real weights
  often lie exactly halfway between two codes, where a quotient that is one bit off
  rounds the other way. Such a flip does not change the size of the error.
- SQNR values are compared to a relative tolerance of `1e-3` (and `1e-3` dB absolute),
  `bits_per_weight` to `1e-6`, and `"worst"` exactly. Compute the sums of squares so
  that they are accurate for a matrix of $10^8$ elements, and count bits in Python
  integers: the real model occupies $3 \cdot 10^9$ of them.
- $2 \le \text{bits} \le 8$. `group_size` divides `in`; MLX itself supports only 32,
  64, and 128, and those are what the tests use, apart from the example. Groups may be
  constant, all zero, or entirely on one side of zero. `weights` holds at least three
  matrices, their SQNR values are distinct, and none is quantized without error.
- The real model has 169 matrices (352.3 M weights) and 57 gain vectors. Group size 64
  at 8 and at 4 bits is tested. Then the quantized model is run by the kit on a
  12-token prompt, and the KL divergence of its next-token distributions from the
  original's, averaged over the positions, must be at most **0.01 nats at 8 bits**
  and **2.5 nats at 4 bits**.
- The time limit per case is 30 seconds. One pass over the real model takes about a
  second.

> [!WARNING]
> In JAX, dividing an array by a Python scalar multiplies by the scalar's reciprocal,
> which is one bit off for about a third of all float32 values. In step 1 that bit can
> turn a tie in $q_0$ and move a whole group to another grid. Divide by an array
> (`jnp.full_like(x, L)`) where the quotient feeds a `round`.

## Why this matters

"4-bit" is a name, not a measurement. The file that `mlx_lm.convert -q` writes for
this model at 4 bits holds 4.5 bits per weight, because every group of 64 weights
carries a 16-bit scale and a 16-bit bias, and a model card that says 4.0 is 12 percent
off on the one number people size their memory by. The error side is just as easy to
misreport. Averaging decibels over layers gives a 16,000-weight projection the same
vote as a 100-million-weight embedding; leaving the embedding out of the loop reports
on 71 percent of the model. And the per-layer view is how quantization failures are
actually diagnosed: Granite 350M at 8 bits moves its predictions by 0.003 nats, at 4
bits by 1.5 nats, which is a broken model, although every layer still reports a
respectable 20 dB. Which layers to keep at higher precision is read off exactly the
table this function builds.
