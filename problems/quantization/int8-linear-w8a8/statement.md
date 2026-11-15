Run a linear layer in integer arithmetic: the weights are int8 codes with one scale per
output row, the activations are quantized to int8 on the fly with one affine grid per
token, the products are accumulated in int32, and real numbers appear only in the final
rescale. Each of the four steps is short. The layer is wrong if the accumulator is too
narrow, if the grid is shared across tokens, if the zero-point is forgotten in the
product, or if the bias is added on the wrong side of the rescale.

## Specification

```python
QMIN, QMAX = -128, 127


def int8_linear(x, w_q, w_scale, bias): ...  # -> (y, acc)
```

The layer stands for $y = x \hat W^{\top} + b$ with $\hat W_{ij} = \text{w\_scale}_i
\cdot \text{w\_q}_{ij}$: the weights were quantized symmetrically per output row, as in
`symmetric-per-channel`, and have no zero-point. `x` has shape `(n, in)`; row $r$ is
one token.

**Step 1: quantize the activations, one grid per token.** For every row $r$ of `x`
separately, with the range widened to include 0:

$$
\text{lo}_r = \min\!\big(\min_j x_{rj},\ 0\big), \qquad
\text{hi}_r = \max\!\big(\max_j x_{rj},\ 0\big),
$$

$$
s_r = \begin{cases}
\dfrac{\text{hi}_r - \text{lo}_r}{255} & \text{hi}_r > \text{lo}_r, \\[1.5ex]
1 & \text{the row is all zeros},
\end{cases}
\qquad
z_r = \operatorname{clamp}\!\Big({-128} - \operatorname{round}\!\Big(\frac{\text{lo}_r}{s_r}\Big),\ -128,\ 127\Big),
$$

$$
q_{rj} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{x_{rj}}{s_r}\Big) + z_r,\ -128,\ 127\Big).
$$

This is the affine int8 grid of `affine-int8`, computed per row: the code $q_{rj}$
stands for the real number $s_r \cdot (q_{rj} - z_r)$, and $0.0$ has the code $z_r$.

**Step 2: accumulate in int32.**

$$
\text{acc}_{ri} = \sum_{j} q_{rj} \cdot \text{w\_q}_{ij} .
$$

Every product and the sum are computed in integers at least 32 bits wide. This is the
only place where the layer touches all $\text{in}$ inputs, and it contains no
floating-point number.

**Step 3: correct for the zero-point.** The layer needs
$\sum_j (q_{rj} - z_r)\, \text{w\_q}_{ij}$, not $\text{acc}_{ri}$:

$$
\text{corrected}_{ri} = \text{acc}_{ri} - z_r \cdot \sum_j \text{w\_q}_{ij},
$$

still in integers. The row sums of `w_q` do not depend on `x`.

**Step 4: rescale, then add the bias.**

$$
y_{ri} = s_r \cdot \text{w\_scale}_i \cdot \text{corrected}_{ri} + \text{bias}_i .
$$

- `x`: float32, shape `(n, in)`.
- `w_q`: int8, shape `(out, in)`, with values in $[-127, 127]$.
- `w_scale`: float32, shape `(out,)`, positive; entry $i$ belongs to row $i$ of `w_q`.
- `bias`: float32, shape `(out,)`, in real units (not quantized).
- Returns the pair `(y, acc)`: `y` is float32 of shape `(n, out)`; `acc` is the
  accumulator of step 2, **before** the correction, as an **int32** tensor of shape
  `(n, out)`.

**Rounding.** $\operatorname{round}$ is rounding to the nearest integer, ties to the
even integer, which is what `round` does in PyTorch, JAX, and MLX. No test contains a
value of $x_{rj} / s_r$ or $\text{lo}_r / s_r$ within $0.05$ of a tie, so a one-bit
difference in a float32 division cannot change a code.

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

Token 0 spans $[-0.5, 2.0]$: $s_0 = 2.5 / 255 = 0.0098039$,
$z_0 = -128 - \operatorname{round}(-51) = -77$, and its codes are
$q_0 = [-128, -77, -46, 45, 127]$. Token 1 is non-negative, so its range is widened to
$[0, 5]$: $s_1 = 5 / 255 = 0.0196078$, $z_1 = -128$, and
$q_1 = [-26, 25, 127, -128, -77]$. The two tokens have different grids.

For token 0 and output 0:

$$
\text{acc}_{00} = (-128)(127) + (-77)(-64) + (-46)(0) + (45)(32) + (127)(-127) = -26017 .
$$

The row sum of `w_q[0]` is $-32$, so $\text{corrected}_{00} = -26017 - (-77)(-32) =
-28481$, and
$y_{00} = 0.0098039 \cdot 0.01 \cdot (-28481) + 0.5 = -2.292255$. The float layer on the
dequantized weights gives $-2.291$; the difference is the rounding of $0.3$ and $1.2$
to their codes.

Without the correction, $y_{00}$ would be $0.0098039 \cdot 0.01 \cdot (-26017) + 0.5 =
-2.0507$, and for token 1, whose zero-point is $-128$, the error is larger still:
$-0.15$ becomes $0.6531$.

## Constraints

- $1 \le n \le 64$, $1 \le \text{out} \le 64$, $1 \le \text{in} \le 4096$, and
  $n \cdot \text{out} \cdot \text{in} \le 2{,}000{,}000$.
- Every entry of `x` is finite with $|x_{rj}| \le 10^4$; a row that is not all zeros
  has $\text{hi}_r - \text{lo}_r \ge 10^{-4}$.
- `acc` is compared **exactly** and must have dtype int32. Its magnitude can reach
  $128 \cdot 127 \cdot 4096 \approx 6.7 \cdot 10^{7}$: more than int8 and int16 hold,
  and more than the $2^{24} \approx 1.7 \cdot 10^{7}$ up to which float32 represents
  every integer.
- `y` is compared with the reference to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-5`.
- `y` also stays within the quantization bound of the float layer
  $x \hat W^{\top} + b$: every activation is reproduced within half a step, so
  $\big| y_{ri} - (x \hat W^{\top} + b)_{ri} \big| \le \tfrac{1}{2}\, s_r \sum_j |\hat W_{ij}|$.
- No Python loops over tokens, outputs, or inputs are needed.

## Why this matters

"W8A8" means 8-bit weights and 8-bit activations, and it is what lets a layer run on
integer hardware: the inner loop is a multiply-accumulate of int8 codes into an int32
register, which is what the integer SIMD and NPU instructions provide. The weights
are quantized once, offline. The activations cannot be: their range differs from
token to token, so dynamic schemes such as the `W8A8` preset of llm-compressor compute
a fresh grid for every token at run time. A grid shared across the batch would make
one token's result depend on what else is in the batch. An affine grid adds the
zero-point term, and that term is where integer layers most often go wrong, because a
layer without it still runs, and is nearly right whenever the activations happen to be
centered. The accumulator width fails the same way: int8 products wrap silently, and
a float32 accumulator is exact for narrow layers and off by a few units for wide ones.
