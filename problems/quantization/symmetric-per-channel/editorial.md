## The idea

A symmetric quantizer is an affine quantizer with the zero-point fixed at 0, so $r = \text{scale} \cdot q$. Only one parameter remains per row, set by mapping the row's largest magnitude to the largest code: $\text{scale}_i = \max_j |w_{ij}| / q_{\max}$.

**Why the range is $[-q_{\max}, q_{\max}]$ instead of $[-2^{\,\text{bits}-1}, 2^{\,\text{bits}-1} - 1]$.** Two's-complement integers have one extra negative value. If we scaled $-\max|w|$ to $-2^{\,\text{bits}-1}$, then $+\max|w|$ would require $+2^{\,\text{bits}-1}$, which does not exist and would clip by a full step. Dropping the extra negative code costs only $1/256$ of the range at 8 bits and keeps the grid symmetric, so $-w$ always quantizes to $-q$.

**Why the maximum is taken per row.** Row $i$ of a linear layer's weight matrix produces output $i$. Because the row scale factors out of the dot product,

$$
y_i = \sum_j w_{ij} x_j \approx \text{scale}_i \sum_j q_{ij} x_j,
$$

the hardware can accumulate the integer sum first and multiply by $\text{scale}_i$ once at the end. A per-column scale would sit inside the summation and require a multiplication on every term. Output rows also vary widely in magnitude. If one row has weights near `1.0` and another near `0.01`, a shared scale of $1/127$ crushes the small row into codes `[-1, 0, 1]`, whereas a per-row scale gives it all 255 levels.

Because every entry in row $i$ has magnitude at most $m_i$, the rounded quotient already falls inside $[-q_{\max}, q_{\max}]$. The clamp simply guards against a one-bit float overshoot. Rounding introduces at most half a code of error, so every dequantized weight stays within $\text{scale}_i / 2$ of the original.

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

Reshaping `scale` to `scale[:, None]` gives shape `(out, 1)` so it broadcasts across columns along each row. `dequantize` casts `q` to float32 before multiplying. Unlike affine quantization, there is no zero-point subtraction that could overflow `int8`. The JAX and MLX solutions are direct translations using `jnp.clip` or `mx.clip` and `astype`.

## Common mistakes

- **Using one scale for the whole tensor.** This only matches the row with the global maximum and loses precision on every smaller row.
- **Using the asymmetric two's-complement range.** Dividing by $2^{\,\text{bits}-1}$ and clamping to $[-2^{\,\text{bits}-1}, 2^{\,\text{bits}-1} - 1]$ clips the largest positive weights and makes the grid slightly too fine for every other code.
- **Reducing along the wrong axis.** Taking `amax(dim=0)` computes one scale per input column. Non-square matrices fail with a shape mismatch `(in,)`, but square matrices silently produce wrong results.
- **Broadcasting without the trailing axis.** Writing `w / scale` or `q * scale` with shapes `(out, in)` and `(out,)` aligns `scale` with the columns instead of the rows.
- **Dividing by zero on an all-zero row.** Computing `0 / 0` produces `NaN`, and casting `NaN` to `int8` is framework-dependent. All-zero rows occur in pruned models and unused output channels.
- **Adding a small epsilon to avoid division by zero.** Using `max_abs / qmax + 1e-8` prevents `NaN`, but alters the scale of every row and shifts codes in small-magnitude rows.
- **Using `max` instead of `max` of the absolute value.** A row whose largest-magnitude weight is negative gets a scale that is too small, and an all-negative row gets a negative scale.
- **Ignoring `bits`.** Codes are always stored in `int8`, but the grid bounds depend on `bits` (for example, $[-7, 7]$ for 4-bit weights).

## Complexity

Both functions run in $O(\text{out} \cdot \text{in})$ time, using one reduction pass for the row maxima and one elementwise pass each. The quantized layer stores 1 byte per weight (or less once low-bit codes are packed) plus one float32 scale per output row.

## In the wild

This is the weight format in LiteRT (TensorFlow Lite) 8-bit quantization. Its `CONV_2D`, `DEPTHWISE_CONV_2D`, and `FULLY_CONNECTED` weights use `int8` codes in $[-127, 127]$ with zero-point 0, scaled along the output-channel axis (axis 0 for `CONV_2D` and `FULLY_CONNECTED`, axis 3 for `DEPTHWISE_CONV_2D`). PyTorch's `torch.per_channel_symmetric` with `ch_axis=0` and ONNX's `QuantizeLinear` with an `axis` attribute implement the same scheme. Nagel et al. (2021), "A White Paper on Neural Network Quantization", detail this rationale for pairing symmetric per-channel weights with asymmetric per-tensor activations.

Two details often differ in production. Runtime quantizers typically floor the scale at a small epsilon rather than substituting `1.0` for an all-zero row (the resulting codes are still 0). Also, because a single outlier stretches its row's grid, calibration methods often clip the range to minimize reconstruction error, and LLM formats frequently use finer groups of 32 to 128 weights per scale. The core quantization mechanics remain the same.
