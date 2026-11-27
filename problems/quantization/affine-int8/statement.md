Implement asymmetric (affine) quantization of a float32 tensor to `int8`. You will compute a scale and zero-point from the tensor's value range, map floats to 8-bit integer codes, and map those codes back to float32 while representing `0.0` exactly.

## Specification

```python
QMIN, QMAX = -128, 127


def quant_params(x): ...  # -> (scale, zero_point)
def quantize(x, scale, zero_point): ...  # -> int8 codes
def dequantize(q, scale, zero_point): ...  # -> float32 values
```

An affine quantization grid represents a real number $r$ using an integer code $q \in \{-128, \dots, 127\}$, a float32 step size `scale`, and an integer offset `zero_point`:

$$
r = \text{scale} \cdot (q - \text{zero\_point}).
$$

**`quant_params(x)`** computes one grid for the entire tensor `x` (per-tensor quantization). First, widen the tensor's range to include `0.0`:

$$
\text{lo} = \min(\min x,\ 0), \qquad \text{hi} = \max(\max x,\ 0).
$$

Next, divide the widened span by 255 (the number of intervals between $-128$ and $127$) to get `scale`, defaulting to `1.0` if every element is zero. Then solve for `zero_point` so that `lo` maps to $-128$:

$$
\text{scale} = \begin{cases}
\dfrac{\text{hi} - \text{lo}}{255} & \text{if } \text{hi} > \text{lo}, \\[1.5ex]
1 & \text{if } \text{hi} = \text{lo} \ \text{(every element of } x \text{ is 0)},
\end{cases}
\qquad
\text{zero\_point} = \operatorname{clamp}\!\Big({-128} - \operatorname{round}\!\Big(\frac{\text{lo}}{\text{scale}}\Big),\ -128,\ 127\Big).
$$

Because the range always includes `0.0`, an all-positive tensor is quantized on $[0, \max x]$ and an all-negative tensor on $[\min x, 0]$. As a result, `lo` maps to $-128$, `hi` maps to $127$, and `0.0` maps to `zero_point` exactly.

![The comb marks the grid's levels under a sample of weights. With clipping off, the affine grid spans the lowest weight to the highest and puts a level on 0.](interactive:quant-grid-affine)

**`quantize(x, scale, zero_point)`** maps each float in `x` to an `int8` code:

$$
q = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{x}{\text{scale}}\Big) + \text{zero\_point},\ -128,\ 127\Big).
$$

Values of `x` outside the grid must saturate at $-128$ or $127$ rather than wrap around.

**`dequantize(q, scale, zero_point)`** maps integer codes `q` (with values in $[-128, 127]$) back to float32:

$$
r = \text{scale} \cdot (q - \text{zero\_point}).
$$

When `q` comes from `quantize`, its dtype is `int8`, and `dequantize` must handle `int8` input correctly even though $q - \text{zero\_point}$ spans $[-255, 255]$.

Both `quantize` and `dequantize` receive `scale` and `zero_point` as 0-d tensors matching the output of `quant_params`. Those parameters do not have to come from the same tensor `x`.

**Rounding.** $\operatorname{round}$ rounds to the nearest integer, breaking halfway ties (such as $2.5$) to the nearest even integer. This matches `round` in PyTorch, JAX, and MLX. The tests contain no value of $x / \text{scale}$ or $\text{lo} / \text{scale}$ within $0.05$ of a tie, so a one-bit float32 difference cannot change a code. Truncation and flooring are not rounding and will fail.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | any shape with 1 to 4 dimensions | float32 |
| `scale` | `()` | float32 |
| `zero_point` | `()` | integer (`int32` or `int64`) |
| `q` | same as `x` | `int8` from `quantize`; any integer type into `dequantize` |
| result of `dequantize` | same as `q` | float32 |

## Example

```python
x = [-0.5, 0.0, 0.3, 1.2, 2.0]

scale, zero_point = quant_params(x)  # 0.00980392, -77
q = quantize(x, scale, zero_point)  # [-128, -77, -46, 45, 127]
dequantize(q, scale, zero_point)  # [-0.5, 0.0, 0.303922, 1.196078, 2.0]
```

The range $[-0.5, 2.0]$ already includes `0.0`, so $\text{scale} = 2.5 / 255 = 0.0098039$. That gives $\text{lo} / \text{scale} = -51$ and $\text{zero\_point} = -128 - (-51) = -77$.

| $x$ | $x / \text{scale}$ | rounded | plus zero-point | back to float |
|---|---|---|---|---|
| $-0.5$ | $-51.0$ | $-51$ | $-128$ | $-0.5$ |
| $0.0$ | $0.0$ | $0$ | $-77$ | $0.0$ |
| $0.3$ | $30.6$ | $31$ | $-46$ | $0.303922$ |
| $1.2$ | $122.4$ | $122$ | $45$ | $1.196078$ |
| $2.0$ | $204.0$ | $204$ | $127$ | $2.0$ |

The endpoints and `0.0` reconstruct exactly. Every other value reconstructs within half a step, $\text{scale} / 2 = 0.0049$.

For the all-positive tensor `[2.0, 3.0, 5.0]`, the range widens to $[0, 5]$. This gives $\text{scale} = 5 / 255 = 0.0196078$, $\text{zero\_point} = -128$, and codes `[-26, 25, 127]`. Over a third of the available codes (those below `2.0`) go unused, which is the trade-off for keeping `0.0` on the grid.

## Constraints

- Every entry of `x` is finite with $|x_i| \le 10^6$. A tensor that is not all zeros has $\text{hi} - \text{lo} \ge 10^{-6}$.
- `scale` is compared to a relative tolerance of `1e-4`. `zero_point` and the `int8` codes are compared **exactly**. Dequantized values are compared to a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`, and a dequantized `0.0` must be exactly `0.0`.
- `quantize` must return dtype `int8`.
- In `quantize`, $|x / \text{scale}| \le 10^5$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.

## Why this matters

Eight-bit affine quantization is the standard scheme for integer inference in TensorFlow Lite, ONNX (`QuantizeLinear`), and PyTorch quantized tensors, cutting memory and bandwidth by 4x. The zero-point exists so that `0.0` is represented without error. Zero padding, ReLU outputs, and masked activations produce exact zeros, and a grid that shifts them to `0.004` injects a systematic bias across the entire network.

Small implementation bugs here fail silently. Dividing by 256 instead of 255 leaves the top code unused, casting to `int8` without clamping wraps large positive activations into negative values, and subtracting `zero_point` in `int8` overflows for half of all codes. Every one of these bugs produces a model that still runs while quietly degrading accuracy.
