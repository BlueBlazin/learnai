## The idea

**The quantizer.** Group-wise affine quantization stores a scale, a bias, and one small integer code per weight for each consecutive run of `group_size` weights in a row. The textbook scheme (`groupwise-4bit`) places the first grid point at the group minimum and the last at the group maximum. MLX makes two modifications, both aimed at handling zero accurately.

First, MLX anchors the grid at whichever endpoint is further from zero (the value with the largest magnitude) and steps toward the other endpoint. When the maximum has larger magnitude, the scale is negative. Second, MLX nudges the step size so that zero lands exactly on a grid point. The anchor $e$ sits $e / s$ steps from zero. Rounding that distance to an integer $q_0$ and redefining the scale as $s = e / q_0$ places $e$ exactly $q_0$ steps from zero, so integer code $-q_0$ decodes to $-q_0 \cdot (e / q_0) + e = 0$. Because neural network weights cluster around zero, a grid that misses zero adds noise to most weights and destroys sparsity in pruned matrices.

Snapping zero onto the grid has a trade-off. Whenever $|q_0|$ rounds up, the step shrinks, pushing the opposite end of the range past the maximum code $L = 2^{\text{bits}} - 1$ so it gets clamped. In the worked example, $0.6$ clamps to $0.35$. On real weights this costs roughly 0.15 dB relative to the textbook grid, which is the price MLX pays for an exact zero.

**The report.** Both error pooling and bit accounting reduce to running sums. Signal power and noise power add across matrices, so the pooled model SQNR is the ratio of the two totals. Bit counts add the same way: `bits` per quantized weight, 32 bits of metadata per group, and 16 bits per unquantized element. With groups of 64, the metadata adds exactly 0.5 bits per weight, which is why an 8-bit conversion of this model takes 8.501 bits per weight.

## Reference solution (PyTorch)

```python
def quantize_dequantize(w, bits, group_size):
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1, keepdim=True)
    hi = groups.amax(dim=-1, keepdim=True)
    from_low = lo.abs() > hi.abs()
    step = torch.clamp((hi - lo) / levels, min=1e-7)
    scale = torch.where(from_low, step, -step)
    edge = torch.where(from_low, lo, hi)
    q0 = torch.round(edge / scale)
    scale = torch.where(q0 != 0, edge / torch.where(q0 != 0, q0, 1.0), scale)
    bias = torch.where(q0 != 0, edge, 0.0)
    codes = torch.clamp(torch.round((groups - bias) / scale), 0, levels)
    return (codes * scale + bias).reshape(out_dim, in_dim)


def quantize_report(weights, bits, group_size):
    sqnr_db = {}
    signal_total = noise_total = 0.0
    stored_bits = parameters = 0
    for name in weights:
        w = weights[name]
        parameters += w.numel()
        if w.ndim != 2:
            stored_bits += FLOAT_BITS * w.numel()
            continue
        w_hat = quantize_dequantize(w, bits, group_size)
        signal = float((w * w).sum(dim=-1).sum())
        noise = float(((w - w_hat) ** 2).sum(dim=-1).sum())
        sqnr_db[name] = 10.0 * math.log10(signal / noise)
        signal_total += signal
        noise_total += noise
        stored_bits += bits * w.numel() + 2 * FLOAT_BITS * (w.numel() // group_size)
    return {
        "sqnr_db": sqnr_db,
        "model_sqnr_db": 10.0 * math.log10(signal_total / noise_total),
        "bits_per_weight": stored_bits / parameters,
        "worst": sorted(sqnr_db, key=sqnr_db.get)[:3],
    }
```

The MLX implementation is a direct translation. The JAX implementation divides by `jnp.full_like(hi, levels)` instead of the Python scalar `levels`. JAX compiles division by a scalar as multiplication by its reciprocal, which differs from IEEE division in the last bit for about a third of float32 values and can flip rounding ties in $q_0$.

## Common mistakes

- **Using the textbook grid.** Quantizing from minimum to maximum without MLX's anchor and adjusted scale produces an SQNR about 0.15 dB *higher*, which does not match what `mlx_lm.convert` actually writes.
- **Always anchoring at the minimum.** Roughly half of all groups have a larger magnitude at the maximum and require a negative scale.
- **Skipping the scale adjustment.** Picking the right anchor without recomputing `scale = edge / q0` leaves zero off the grid.
- **Omitting the step floor.** A constant group (such as a zero row in a pruned matrix or an untrained embedding row) has range 0, causing division by zero and NaN outputs.
- **Truncating instead of rounding.** Truncation doubles the maximum quantization error compared to round-to-nearest.
- **Omitting the clamp.** After adjusting the scale, some weights map above code $L$. Without clamping, those weights reconstruct better than any stored file can represent, inflating the reported SQNR.
- **Strided grouping.** Reshaping to `(out, group_size, n_groups)` groups every `n_groups`-th column instead of consecutive runs of `group_size` inputs.
- **Ignoring scale and bias overhead in `bits_per_weight`.** Reporting 4.0 instead of 4.5 misses the 32 bits of metadata per group.
- **Skipping the embedding matrix.** Filtering tensors by layer name instead of dimensionality (`w.ndim == 2`) leaves out the embedding matrix.
- **Excluding unquantized tensors from `bits_per_weight`.** Norm gains are tiny in Granite 350M (58 thousand of 352 million parameters), but unquantized tensors matter in smaller test models and in architectures with large biases.
- **Averaging per-layer decibels.** Decibels are logarithmic, so their arithmetic mean is a geometric mean that weights every matrix equally regardless of size. Always pool the raw signal and noise powers first.
- **Using `20 * log10` instead of `10 * log10`.** Signal-to-noise ratio is a ratio of squared amplitudes (powers), which uses a factor of 10.
- **Sorting `"worst"` in descending order.** The worst layers are the ones with the *lowest* SQNR.
- **Counting bits in 32-bit integers.** The real model occupies roughly $3.0 \cdot 10^9$ bits, which overflows a signed 32-bit integer and wraps negative after passing every small test. Always accumulate bit counts in Python `int`s.

## Complexity

The algorithm makes one pass over the weights with a small constant number of elementwise operations per weight, holding only one matrix and its dequantized copy in memory at a time. On Granite 4.0 350M, a full pass finishes in under a second.

## In the wild

`mlx_lm.convert -q --q-bits 4 --q-group-size 64` applies this exact scheme to every `nn.Linear` and `nn.Embedding` layer and prints the effective bits per weight computed the same way. On Granite 350M, 8-bit quantization is nearly lossless (0.003 nats KL divergence), while 4-bit quantization degrades output noticeably (1.5 nats KL divergence, changing the greedy token at 3 of 12 positions) despite a respectable ~20 dB SQNR on every layer. Small models have little redundancy, which is why mixed-precision recipes keep embeddings and the first and last layers at 6 or 8 bits, guided by the per-layer SQNR table built here. The next problem, `mlx-quantized-checkpoint-load`, reads the checkpoint format this quantizer produces.
