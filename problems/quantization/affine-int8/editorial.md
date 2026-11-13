## The idea

An affine quantizer is a straight line between floats and integers,
$r = \text{scale} \cdot (q - \text{zero\_point})$, fixed by two requirements:

1. The ends of the (widened) range land on the ends of the code range: `lo` on $-128$
   and `hi` on $127$. There are 255 steps between 256 codes, so
   $\text{scale} = (\text{hi} - \text{lo}) / 255$.
2. `zero_point` is an integer. Then the real number $0$ corresponds to the code
   $q = \text{zero\_point}$ exactly, with no rounding error.

Solving requirement 1 for the zero-point gives $-128 - \text{lo}/\text{scale}$, which
is generally not an integer, so it is rounded. That nudges the grid by at most half a
step: `lo` and `hi` may now be off by up to half a step, in exchange for $0$ being
exact. Widening the range to include $0$ guarantees $\text{lo} \le 0 \le \text{hi}$, so
$-\text{lo}/\text{scale}$ is between 0 and 255 and the zero-point is a valid int8 code.
The clamp only absorbs float rounding at the ends.

Quantizing is the inverse map followed by rounding and saturation; dequantizing is the
line itself. For any value inside the range the round trip is off by at most
$\text{scale} / 2$.

## Reference solution (PyTorch)

```python
QMIN, QMAX = -128, 127


def quant_params(x):
    zero = torch.zeros((), dtype=x.dtype)
    lo = torch.minimum(x.min(), zero)
    hi = torch.maximum(x.max(), zero)
    scale = torch.where(hi > lo, (hi - lo) / (QMAX - QMIN), torch.ones_like(hi))
    zero_point = torch.clamp(QMIN - torch.round(lo / scale), QMIN, QMAX)
    return scale, zero_point.to(torch.int32)


def quantize(x, scale, zero_point):
    q = torch.round(x / scale) + zero_point
    return torch.clamp(q, QMIN, QMAX).to(torch.int8)


def dequantize(q, scale, zero_point):
    return (q.to(torch.int32) - zero_point).to(torch.float32) * scale
```

`quantize` rounds, shifts, and clamps in float32, where integers up to $2^{24}$ are
exact, and converts to int8 last. `dequantize` widens to int32 first. The JAX and MLX
versions are translations with `jnp.clip` / `mx.clip` and `astype`.

## Rounding

`round` in PyTorch, JAX, MLX, and NumPy rounds halves to the nearest even integer
(banker's rounding), the IEEE 754 default: `round(2.5) = 2`, `round(3.5) = 4`. C's
`round` and the `floor(x + 0.5)` idiom round halves away from zero or up. The choices
differ only on exact ties, and a tie in $x / \text{scale}$ is fragile anyway: whether
a float32 quotient lands exactly on $.5$ depends on the last bit of `scale`. That is why
the tests keep clear of ties, and why a quantized model exported from one runtime to
another can differ by one code in a handful of weights. What is never acceptable is
truncation, the default of a float-to-int cast, which is wrong by one code for half of
all values and biases every value toward zero.

## Common mistakes

- **Not widening the range to include 0.** For an all-positive tensor the raw range
  gives a finer grid, and a zero-point far outside $[-128, 127]$. After clamping it,
  $0.0$ and everything near it is misrepresented.
- **Dividing by 256.** The number of codes is 256; the number of steps is 255. With
  256, `hi` would need code $128$, which does not exist: the top of the range
  saturates and every other value is off by up to one code.
- **The sign of the zero-point.** $-128 + \text{lo}/\text{scale}$ instead of
  $-128 - \text{lo}/\text{scale}$. It is right when `lo` is 0, that is for every
  all-positive tensor, such as the output of a ReLU.
- **Flooring the zero-point** (or casting it to int, which truncates). The grid is
  shifted by a whole code for about half of all ranges.
- **Truncating instead of rounding** in `quantize`.
- **Casting without clamping.** In PyTorch on this platform, `torch.tensor(300.0).to(torch.int8)`
  is `44`: the value wraps. JAX and MLX happen to saturate on the cast, which makes
  this a bug that appears when code is ported. An out-of-range float-to-int cast is
  undefined behavior in C, so no result is guaranteed. Clamp first.
- **Subtracting the zero-point in int8.** $q - \text{zero\_point}$ spans $[-255, 255]$.
  In PyTorch an int8 tensor minus a 0-d integer tensor is still int8, so
  `127 - (-128)` wraps to `-1`.

## Complexity

$O(n)$ time: one pass for the minimum and maximum, one elementwise pass for each of
`quantize` and `dequantize`. The quantized tensor takes a quarter of the memory of the
float32 original, plus two scalars.

## In the wild

This is the quantization scheme of Jacob et al. (2018), "Quantization and Training of
Neural Networks for Efficient Integer-Arithmetic-Only Inference", used by TensorFlow
Lite, and the same formulas appear in PyTorch's `MinMaxObserver` (which widens the
range with `min(min_val, 0)` and `max(max_val, 0)`, and floors the scale at a small
epsilon where this problem returns 1) and in ONNX's `QuantizeLinear`, which specifies
`saturate(round(x / scale) + zero_point)` with round-half-to-even.

Real systems refine it in three directions. **Granularity**: one scale per output
channel, or per small block of weights, instead of per tensor, because one outlier
otherwise stretches the grid for everything. **Symmetry**: weights are usually
quantized symmetrically (zero-point 0, range $\pm\max|x|$), which makes integer matrix
multiplication cheaper because the zero-point cross terms vanish; the asymmetric form
of this problem is used for activations, whose ranges are lopsided (after a ReLU,
entirely non-negative). **Range selection**: the minimum and maximum are sensitive to
outliers, so calibration often clips to a percentile or minimizes the quantization
error instead. LLM weight formats mostly use block-wise symmetric or low-bit schemes,
but the mechanics you wrote here, scale, rounding, clamping, and widening before
integer arithmetic, are the same in all of them.
