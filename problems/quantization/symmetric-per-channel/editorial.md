## The idea

A symmetric quantizer is the affine quantizer of `affine-int8` with the zero-point
pinned to 0: $r = \text{scale} \cdot q$. One parameter is left, and one requirement
fixes it: the largest magnitude of the row lands on the largest code,
$\text{scale} = \max |w| / q_{\max}$.

Two decisions are packed into that sentence.

**The range is $[-q_{\max}, q_{\max}]$, not $[-2^{b-1}, 2^{b-1} - 1]$.** Two's
complement has one more negative code than positive ones. If the scale were chosen so
that $-\max|w|$ maps to $-2^{b-1}$, then $+\max|w|$ would need the code $+2^{b-1}$,
which does not exist, and would be clipped: the largest positive weight of every row
would be off by a full step. Dropping the extra negative code costs $1/256$ of the
range at 8 bits and makes the grid symmetric, so that $-w$ quantizes to exactly $-q$.

**The maximum is taken per row.** Row $i$ of a linear layer's weight produces output
$i$, and nothing in the layer mixes the scales of different outputs:

$$
y_i = \sum_j w_{ij} x_j \approx \text{scale}_i \sum_j q_{ij} x_j .
$$

The integer sum is computed first and the row's scale multiplies it once at the end,
so a scale per row is free at inference time. A scale per *column* would not be: it
would sit inside the sum, one multiplication per term. Rows are also where the
magnitudes differ. If one row has weights near 1 and another near 0.01, a shared
scale of $1/127$ gives the small row the codes $-1, 0, 1$ and a relative error of
tens of percent, while its own scale gives it the full 255 levels.

Because every element of a row has magnitude at most $m_i$, the rounded quotient
already lies in $[-q_{\max}, q_{\max}]$ and the clamp only absorbs a possible last-bit
overshoot of the division. Inside the range, rounding is off by at most half a code,
so the round-trip error is at most $\text{scale}_i / 2$.

## Reference solution (PyTorch)

```python
def quantize_weight(w, bits):
    qmax = 2 ** (bits - 1) - 1
    max_abs = w.abs().amax(dim=1)
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(w / scale[:, None]), -qmax, qmax)
    return q.to(torch.int8), scale


def dequantize(q, scale):
    return q.to(torch.float32) * scale[:, None]
```

`scale[:, None]` has shape `(out, 1)` and broadcasts along each row. The codes are
converted to float32 before the multiplication; there is no subtraction of a
zero-point, so unlike the affine case nothing can overflow int8. The JAX and MLX
versions are translations with `jnp.clip` / `mx.clip` and `astype`.

## Common mistakes

- **One scale for the whole tensor.** Correct for the row that holds the global
  maximum and increasingly wrong for every smaller row.
- **The asymmetric range.** Dividing by $2^{b-1}$ and clamping to
  $[-2^{b-1}, 2^{b-1} - 1]$: the largest positive weight is clipped, and every other
  code is computed on a grid that is slightly too fine.
- **The wrong axis.** `amax(dim=0)` gives one scale per input column. For a
  non-square matrix the shape `(in,)` gives it away; for a square one nothing does.
- **Broadcasting without the extra axis**, in either function. `q * scale` with
  shapes `(out, in)` and `(out,)` aligns the scales with the columns.
- **Dividing by zero for an all-zero row.** $0 / 0$ is NaN, and what a NaN becomes
  when cast to int8 depends on the framework. Rows of zeros are not exotic: pruned
  models and unused output channels have them.
- **Hiding the division by zero with an epsilon.** `max_abs / qmax + 1e-8` avoids the
  NaN, and changes the scale of every row; for rows of small weights the change is
  large enough to move codes.
- **`max` instead of `max` of the absolute value.** A row whose extreme weight is
  negative gets a scale that is too small, and a row of negative weights a negative
  one.
- **Ignoring `bits`.** The codes are always stored in int8, but the range is set by
  `bits`: 4-bit codes live in $[-7, 7]$.

## Complexity

$O(\text{out} \cdot \text{in})$ time: one pass for the row maxima and one elementwise
pass for each function. The quantized layer stores one byte per weight (or less, once
low-bit codes are packed) plus one float per row.

## In the wild

This is the weight scheme of the LiteRT (TensorFlow Lite) 8-bit quantization
specification: weights of `CONV_2D`, `DEPTHWISE_CONV_2D`, and `FULLY_CONNECTED` are
int8 in the range $[-127, 127]$ with zero-point 0, quantized per-axis along the
output-channel dimension (dimension 0 for `CONV_2D` and `FULLY_CONNECTED`, dimension 3
for `DEPTHWISE_CONV_2D`). PyTorch's qscheme `torch.per_channel_symmetric` with
`ch_axis=0` is the same idea, and ONNX's `QuantizeLinear` expresses it with an `axis`
attribute and a vector of scales. Nagel et
al. (2021), "A White Paper on Neural Network Quantization", give the reasoning used
above for why weights are symmetric and per-channel while activations are asymmetric
and per-tensor.

Two things differ in practice. Real quantizers floor the scale at a small epsilon
instead of substituting 1 for an all-zero row; the codes are 0 either way. And the
maximum is a crude range: a single outlier in a row stretches that row's grid, so
calibration methods clip the range to minimize the error instead, and LLM formats go
one step finer than a row, with one scale per group of 32 to 128 consecutive weights.
Both refinements keep the mechanics you wrote here.
