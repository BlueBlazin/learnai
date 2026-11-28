Implement an integer-only linear layer (W8A8) that quantizes float32 activations to int8 per token, multiplies them with pre-quantized int8 weights in int32, and rescales the result to float32 at the end. You will handle per-token affine zero-points, exact int32 accumulation, and post-rescale bias addition.

## Specification

```python
QMIN, QMAX = -128, 127


def int8_linear(x, w_q, w_scale, bias): ...  # -> (y, acc)
```

The layer approximates $y = x \hat W^{\top} + b$ for a batch of $n$ tokens `x` (where row $r$ is token $r$) and a float32 `bias` vector $b$ in real units. The weights are already quantized symmetrically per output row, as in `symmetric-per-channel`, with no zero-point:

$$
\hat W_{ij} = \text{w\_scale}_i \cdot \text{w\_q}_{ij},
$$

where `w_q` has integer values in $[-127, 127]$ and `w_scale` has one positive float32 scale per output row.

Perform the computation in four steps.

**Step 1: quantize the activations with one affine grid per token.** For each row $r$ of `x` independently, widen the row's range to include $0$:

$$
\text{lo}_r = \min\!\big(\min_j x_{rj},\ 0\big), \qquad
\text{hi}_r = \max\!\big(\max_j x_{rj},\ 0\big).
$$

Compute the row scale $s_r$, the zero-point $z_r$, and the int8 activation codes $q_{rj}$:

$$
s_r = \begin{cases}
\dfrac{\text{hi}_r - \text{lo}_r}{255} & \text{if } \text{hi}_r > \text{lo}_r, \\[1.5ex]
1 & \text{if the row is all zeros},
\end{cases}
\qquad
z_r = \operatorname{clamp}\!\Big({-128} - \operatorname{round}\!\Big(\frac{\text{lo}_r}{s_r}\Big),\ -128,\ 127\Big),
$$

$$
q_{rj} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{x_{rj}}{s_r}\Big) + z_r,\ -128,\ 127\Big).
$$

This is the per-row affine int8 grid from `affine-int8`. Each code $q_{rj}$ represents the real value $s_r \cdot (q_{rj} - z_r)$, and $0.0$ maps to $z_r$.

**Step 2: accumulate the integer matrix product in int32.**

$$
\text{acc}_{ri} = \sum_{j} q_{rj} \cdot \text{w\_q}_{ij}.
$$

Compute every product and the sum using integers at least 32 bits wide, with no floating-point arithmetic.

**Step 3: subtract the zero-point contribution.** The layer requires $\sum_j (q_{rj} - z_r)\, \text{w\_q}_{ij}$ rather than $\text{acc}_{ri}$. Apply this correction in integers:

$$
\text{corrected}_{ri} = \text{acc}_{ri} - z_r \cdot \sum_j \text{w\_q}_{ij}.
$$

Notice that the row sums of `w_q` do not depend on `x`.

**Step 4: rescale to float32, then add the bias.**

$$
y_{ri} = s_r \cdot \text{w\_scale}_i \cdot \text{corrected}_{ri} + \text{bias}_i.
$$

Return the pair `(y, acc)`, where `y` is the float32 output and `acc` is the int32 accumulator from step 2 **before** the zero-point correction.

**Rounding.** Here $\operatorname{round}$ rounds to the nearest integer with ties going to the even integer, which matches `round` in PyTorch, JAX, and MLX. No test has $x_{rj} / s_r$ or $\text{lo}_r / s_r$ within $0.05$ of a tie, so float32 division differences cannot change any code.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(n, in)` | float32 |
| `w_q` | `(out, in)` | int8 |
| `w_scale` | `(out,)` | float32 |
| `bias` | `(out,)` | float32 |
| `y` | `(n, out)` | float32 |
| `acc` | `(n, out)` | int32 |

## Example

```python
x = [
    [-0.5, 0.0, 0.3, 1.2, 2.0],
    [2.0, 3.0, 5.0, 0.0, 1.0],
]
w_q = [
    [127, -64, 0, 32, -127],
    [10, 20, 30, 40, 50],
]
w_scale = [0.01, 0.002]
bias = [0.5, -1.0]

y, acc = int8_linear(x, w_q, w_scale, bias)
# acc = [[-26017,  3950],
#        [   781, -4920]]
# y = [[-2.292255, -0.696078],
#      [-0.15,     -0.44    ]]
```

Token 0 spans $[-0.5, 2.0]$. Its scale is $s_0 = 2.5 / 255 = 0.0098039$, its zero-point is $z_0 = -128 - \operatorname{round}(-51) = -77$, and its codes are $q_0 = [-128, -77, -46, 45, 127]$. Token 1 is non-negative, so its range widens to $[0, 5]$, giving $s_1 = 5 / 255 = 0.0196078$, $z_1 = -128$, and $q_1 = [-26, 25, 127, -128, -77]$. Each token uses its own grid.

For token 0 and output 0, the int32 accumulator is:

$$
\text{acc}_{00} = (-128)(127) + (-77)(-64) + (-46)(0) + (45)(32) + (127)(-127) = -26017.
$$

The row sum of `w_q[0]` is $-32$, so $\text{corrected}_{00} = -26017 - (-77)(-32) = -28481$. Rescaling and adding the bias gives $y_{00} = 0.0098039 \cdot 0.01 \cdot (-28481) + 0.5 = -2.292255$. The float layer on the dequantized weights produces $-2.291$, with the small difference coming from rounding $0.3$ and $1.2$ to their int8 codes.

Without the zero-point correction, $y_{00}$ would be $0.0098039 \cdot 0.01 \cdot (-26017) + 0.5 = -2.0507$. For token 1, whose zero-point is $-128$, omitting the correction causes an even larger error, turning $-0.15$ into $0.6531$.

## Constraints

- $1 \le n \le 64$, $1 \le \text{out} \le 64$, $1 \le \text{in} \le 4096$, and $n \cdot \text{out} \cdot \text{in} \le 2{,}000{,}000$.
- Every entry of `x` is finite with $|x_{rj}| \le 10^4$, and every row that is not all zeros has $\text{hi}_r - \text{lo}_r \ge 10^{-4}$.
- `acc` is compared **exactly** and must have dtype `int32`. Its magnitude can reach $128 \cdot 127 \cdot 4096 \approx 6.7 \cdot 10^7$, which overflows `int8` and `int16` and exceeds the $2^{24} \approx 1.7 \cdot 10^7$ limit up to which float32 represents every integer exactly.
- `y` is compared against the reference with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`.
- `y` also satisfies the quantization error bound of the float layer $x \hat W^{\top} + b$. Because every activation is represented within half a grid step, $\big| y_{ri} - (x \hat W^{\top} + b)_{ri} \big| \le \tfrac{1}{2}\, s_r \sum_j |\hat W_{ij}|$.
- No Python loops over tokens, outputs, or inputs are needed.

## Why this matters

"W8A8" stands for 8-bit weights and 8-bit activations. It lets a linear layer run on integer SIMD and NPU instructions whose inner loop multiplies int8 codes and accumulates into an int32 register. Weights are quantized once offline, while activations vary from token to token, so dynamic schemes such as the `W8A8` preset in `llm-compressor` compute a fresh grid per token at run time. Sharing one grid across a batch would make each token's output depend on the other tokens in the batch.

Two bugs are especially hard to catch in practice. Omitting the zero-point correction still produces plausible outputs whenever activations happen to be centered near zero, failing only on skewed activations. Using an accumulator that is too narrow fails silently too. Int8 products wrap around, while a float32 accumulator stays exact on narrow layers and drifts by a few units on wide ones.
