Implement asymmetric (affine) quantization of a float32 tensor to int8: choose a scale
and a zero-point from the tensor's range, map floats to 8-bit codes, and map codes back.
The arithmetic is short, but every step has an off-by-one, a rounding choice, or an
integer overflow waiting in it, and the grid must contain 0.0 exactly.

## Specification

```python
QMIN, QMAX = -128, 127


def quant_params(x): ...  # -> (scale, zero_point)
def quantize(x, scale, zero_point): ...  # -> int8 codes
def dequantize(q, scale, zero_point): ...  # -> float32 values
```

An affine grid represents the real number $r$ by the integer code $q$ with

$$
r = \text{scale} \cdot (q - \text{zero\_point}), \qquad q \in \{-128, \dots, 127\}.
$$

**`quant_params(x)`** computes one grid for the whole tensor (per-tensor quantization):

$$
\text{lo} = \min(\min x,\ 0), \qquad \text{hi} = \max(\max x,\ 0),
$$

$$
\text{scale} = \begin{cases}
\dfrac{\text{hi} - \text{lo}}{255} & \text{hi} > \text{lo}, \\[1.5ex]
1 & \text{hi} = \text{lo} \ \text{(every element of } x \text{ is 0)},
\end{cases}
\qquad
\text{zero\_point} = \operatorname{clamp}\!\Big({-128} - \operatorname{round}\!\Big(\frac{\text{lo}}{\text{scale}}\Big),\ -128,\ 127\Big).
$$

The range is widened to include 0 before anything else. An all-positive tensor is
quantized on $[0, \max x]$ and an all-negative one on $[\min x, 0]$. As a result
`lo` maps to code $-128$, `hi` maps to code $127$, and $0.0$ maps to the code
`zero_point` exactly.

It returns `scale` as a 0-d float32 tensor and `zero_point` as a 0-d integer tensor.

**`quantize(x, scale, zero_point)`**:

$$
q = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{x}{\text{scale}}\Big) + \text{zero\_point},\ -128,\ 127\Big),
$$

returned as an **int8** tensor of the same shape as `x`. Values of `x` beyond the grid
saturate at $-128$ or $127$; they must not wrap around.

**`dequantize(q, scale, zero_point)`**:

$$
r = \text{scale} \cdot (q - \text{zero\_point}),
$$

returned as a float32 tensor of the same shape as `q`. `q` is an integer tensor with
values in $[-128, 127]$; it is int8 when it comes from `quantize`, and the function
must be correct for int8 input. Note that $q - \text{zero\_point}$ ranges over
$[-255, 255]$.

**Rounding.** $\operatorname{round}$ is rounding to the nearest integer. Ties (values
exactly halfway, like $2.5$) go to the even integer, which is what `round` does in
PyTorch, JAX, and MLX. The tests contain no value of $x / \text{scale}$ or
$\text{lo} / \text{scale}$ within $0.05$ of a tie, so a one-bit difference in a float32
division cannot change a code. Truncation and flooring are not rounding and do fail.

`quantize` and `dequantize` receive `scale` and `zero_point` as 0-d tensors, exactly as
`quant_params` returns them, and do not have to come from the same tensor `x`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | any shape with one to four dimensions | float32 |
| `scale` | `()` | float32 |
| `zero_point` | `()` | integer (int32 or int64) |
| `q` | same as `x` | int8 from `quantize`; any integer type into `dequantize` |
| result of `dequantize` | same as `q` | float32 |

## Example

```python
x = [-0.5, 0.0, 0.3, 1.2, 2.0]

scale, zero_point = quant_params(x)  # 0.00980392, -77
q = quantize(x, scale, zero_point)  # [-128, -77, -46, 45, 127]
dequantize(q, scale, zero_point)  # [-0.5, 0.0, 0.303922, 1.196078, 2.0]
```

The range is $[-0.5, 2.0]$, so $\text{scale} = 2.5 / 255 = 0.0098039$. Then
$\text{lo} / \text{scale} = -51$ and $\text{zero\_point} = -128 + 51 = -77$.

| $x$ | $x / \text{scale}$ | rounded | plus zero-point | back to float |
|---|---|---|---|---|
| $-0.5$ | $-51.0$ | $-51$ | $-128$ | $-0.5$ |
| $0.0$ | $0.0$ | $0$ | $-77$ | $0.0$ |
| $0.3$ | $30.6$ | $31$ | $-46$ | $0.303922$ |
| $1.2$ | $122.4$ | $122$ | $45$ | $1.196078$ |
| $2.0$ | $204.0$ | $204$ | $127$ | $2.0$ |

The ends of the range and $0.0$ come back exactly. Everything else comes back within
half a step, $\text{scale} / 2 = 0.0049$.

For the all-positive tensor `[2.0, 3.0, 5.0]` the range is widened to $[0, 5]$:
$\text{scale} = 5 / 255 = 0.0196078$, $\text{zero\_point} = -128$, and the codes are
`[-26, 25, 127]`. More than a third of the codes (those for values below 2) are unused.
That is the price of keeping $0.0$ on the grid.

## Constraints

- Every entry of `x` is finite with $|x_i| \le 10^6$. A tensor that is not all zeros
  has $\text{hi} - \text{lo} \ge 10^{-6}$.
- `scale` is compared to a relative tolerance of `1e-4`. `zero_point` and the int8
  codes are compared **exactly**. Dequantized values are compared to a relative
  tolerance of `1e-4` and an absolute tolerance of `1e-6`, and a dequantized $0.0$
  must be exactly $0.0$.
- `quantize` must return dtype int8.
- In `quantize`, $|x / \text{scale}| \le 10^5$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.

## Why this matters

Eight-bit affine quantization is the workhorse of integer inference: it is the scheme
of TensorFlow Lite, of ONNX's `QuantizeLinear`, and of PyTorch's quantized tensors, and
it cuts memory and bandwidth by four. The zero-point exists for one reason, which is
that $0.0$ must be exactly representable. Zero padding, ReLU outputs, and masked
activations are exact zeros, and a grid that turns them into $0.004$ adds a bias to
every one of them. The rest is details that decide whether a quantized model matches
its float original: a scale computed with 256 instead of 255 leaves the top code
unused; a cast without a clamp turns the largest activation into a large negative one;
and subtracting the zero-point in int8 overflows for half of all codes. Each of these
produces a model that runs and is quietly worse.
