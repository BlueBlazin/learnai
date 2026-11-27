## The idea

An affine quantizer defines a linear map between floats and integers, $r = \text{scale} \cdot (q - \text{zero\_point})$, pinned down by two requirements:

1. The endpoints of the widened range land on the endpoints of the `int8` range (`lo` on $-128$ and `hi` on $127$). Because 256 codes have 255 intervals between them, $\text{scale} = (\text{hi} - \text{lo}) / 255$.
2. `zero_point` is an integer, so `0.0` corresponds to the code $q = \text{zero\_point}$ with zero rounding error.

Solving requirement 1 for `zero_point` yields $-128 - \text{lo} / \text{scale}$. That value is rarely an exact integer, so we round it. Rounding shifts the grid by at most half a step, trading a tiny endpoint error for an exact representation of `0.0`. Widening the range to include `0.0` ensures $\text{lo} \le 0 \le \text{hi}$, which keeps $-\text{lo} / \text{scale}$ in $[0, 255]$ and keeps `zero_point` within $[-128, 127]$. The clamp only guards against floating-point rounding at the boundaries.

Quantizing applies the inverse map, rounds to the nearest integer, and saturates to $[-128, 127]$. Dequantizing evaluates the linear map directly. For any value inside $[\text{lo}, \text{hi}]$, the round-trip error is at most $\text{scale} / 2$.

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

`quantize` performs rounding, shifting, and clamping in float32, where integers up to $2^{24}$ are exact, and casts to `int8` last. `dequantize` widens `q` to `int32` before subtracting `zero_point`. The JAX and MLX solutions follow the same structure using `jnp.clip` or `mx.clip` and `astype`.

## Rounding

`round` in PyTorch, JAX, MLX, and NumPy uses round-half-to-even (banker's rounding), the IEEE 754 default, so `round(2.5) = 2` and `round(3.5) = 4`. C's `round` and `floor(x + 0.5)` instead round halves away from zero or upward. These modes only differ on exact `.5` ties, which are already sensitive to the last bit of `scale` in float32. That is why the test suite avoids values near ties, and why exporting a quantized model across runtimes can shift a few weights by one code. Truncation (the default behavior of casting float to int) is never acceptable because it is off by one code for half of all values and biases every value toward zero.

## Common mistakes

- **Not widening the range to include 0.** Using the raw range of an all-positive tensor produces a finer grid, but pushes `zero_point` outside $[-128, 127]$. After clamping, `0.0` and nearby values are badly misrepresented.
- **Dividing by 256.** There are 256 codes, but only 255 steps between them. Dividing by 256 maps `hi` to $128$, which does not fit in `int8`, saturating the top of the range and shifting other codes by up to one step.
- **Wrong sign on `zero_point`.** Writing $-128 + \text{lo} / \text{scale}$ instead of $-128 - \text{lo} / \text{scale}$ happens to work whenever `lo` is `0.0` (such as after ReLU), hiding the bug on non-negative tensors.
- **Flooring or truncating `zero_point`.** Casting to int without rounding shifts the entire grid by one code for about half of all input ranges.
- **Truncating instead of rounding in `quantize`.** Casting directly to an integer type drops the fractional part instead of rounding to the nearest code.
- **Casting without clamping.** In PyTorch on this platform, `torch.tensor(300.0).to(torch.int8)` wraps to `44`. JAX and MLX saturate on the cast, so this bug can remain hidden until code is ported. Out-of-range float-to-int conversion is undefined behavior in C, so always clamp before casting.
- **Subtracting `zero_point` in `int8`.** The difference $q - \text{zero\_point}$ ranges over $[-255, 255]$. In PyTorch, subtracting a 0-d integer tensor from an `int8` tensor keeps the result in `int8`, so `127 - (-128)` wraps to `-1`.

## Complexity

$O(n)$ time for an $n$-element tensor. Computing the range takes one reduction pass, and `quantize` and `dequantize` each take one elementwise pass. The quantized tensor uses one-fourth the memory of float32, plus two scalars.

## In the wild

This scheme comes from Jacob et al. (2018), "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", and underpins TensorFlow Lite. PyTorch's `MinMaxObserver` uses the same formulas (widening with `min(min_val, 0)` and `max(max_val, 0)`, and flooring the scale at a small epsilon instead of `1.0`), as does ONNX's `QuantizeLinear` with round-half-to-even.

Production systems extend this baseline in three main ways:

- **Granularity:** Assigning one scale per output channel or per small block of weights prevents a single outlier from stretching the grid for the whole tensor.
- **Symmetry:** Weights are usually quantized symmetrically (`zero_point = 0`, range $\pm \max|x|$) so integer matrix multiplication avoids extra zero-point cross terms. Asymmetric quantization is kept for activations, whose distributions are often lopsided (such as non-negative ReLU outputs).
- **Range selection:** Because raw min/max values are sensitive to outliers, calibration pipelines often clip to percentiles or minimize quantization error directly.

Modern LLM weight formats rely on block-wise symmetric or low-bit grids, but the core operations (scaling, rounding, clamping, and widening before integer arithmetic) remain the same.
