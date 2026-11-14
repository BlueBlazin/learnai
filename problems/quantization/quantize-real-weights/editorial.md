## The idea

**The quantizer.** Group-wise affine quantization stores, for each run of
`group_size` weights in a row, a scale, a bias, and one small integer per weight. The
textbook choice (`groupwise-4bit`) puts the grid's first point on the minimum and its
last on the maximum. MLX makes two different choices, and both are about zero.

First, the grid is anchored at the end of the range that is *further* from zero, the
value that would cost the most to get wrong, and counts toward the other end; that is
why the scale can be negative. Second, the step is nudged so that zero is a grid
point. The anchor $e$ lies $e / s$ steps from zero. That number is rounded to an
integer $q_0$, and the scale is redefined as $e / q_0$: now $e$ is exactly $q_0$ steps
from zero, so code $-q_0$ decodes to $-q_0 \cdot e / q_0 + e = 0$. Weights are
centered on zero and many are small, so a grid without a zero adds noise to every one
of them, and a sparse or pruned matrix would stop being sparse.

The nudge is not free: if $|q_0|$ was rounded up, the step shrank, and the far end of
the range now lies beyond code $L$ and is clamped. In the worked example, $0.6$ comes
back as $0.35$. On real weights this costs about 0.15 dB against the textbook grid,
which is the price MLX pays for an exact zero.

**The report.** Everything is a sum. Signal and noise power add up across matrices,
so the model's SQNR is the ratio of the totals. Bits add up as well: codes, plus 32
bits per group, plus 16 per element for what stays in floating point. With groups of
64 the overhead is exactly half a bit per weight, which is why an "8-bit" conversion
of this model is reported as 8.501 bits per weight.

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

The MLX version is a translation. The JAX version divides by
`jnp.full_like(hi, levels)` instead of by `levels`: JAX turns division by a scalar
into multiplication by its reciprocal, which differs from true division in the last
bit for a third of all values, and the last bit of the step decides ties in $q_0$.

## Common mistakes

- **The textbook grid.** Minimum and range instead of MLX's edge and adjusted scale.
  The SQNR comes out 0.15 dB *better*, which is how you can tell it is not what the
  converter does: the files will not match.
- **Always starting at the minimum.** Half the groups have their larger magnitude at
  the maximum; those need the negative scale.
- **No adjustment of the scale.** The right edge, the textbook step, and no zero on
  the grid.
- **No floor on the step.** A constant group (a zero row of a pruned matrix, an
  untrained embedding row) has range 0, scale 0, and decodes to NaN.
- **Truncating instead of rounding**, which doubles the noise amplitude.
- **No clamp.** After the adjustment some weights need a code above $L$. Unclamped
  they are reproduced better than any file can store them, and the report flatters
  the quantizer.
- **Strided groups.** A reshape to `(out, group_size, n_groups)` groups every
  `n_groups`-th input. Groups are consecutive runs.
- **Bits per weight without the overhead.** 4.0 instead of 4.5.
- **Skipping the embedding**, or anything else whose name does not look like a layer.
  The rule is the shape: two dimensions.
- **Leaving the unquantized tensors out of bits per weight.** They are small here
  (58 thousand of 352 million parameters) and not small in a model with large biases
  or in a small test model.
- **The mean of the decibels.** Decibels are logarithms; their mean is a geometric
  mean of ratios in which every matrix has the same weight. Pool the powers.
- **20 instead of 10.** A ratio of powers takes $10 \log_{10}$.
- **Sorting the wrong way.** The worst layers have the *lowest* SQNR.
- **Counting bits in 32 bits.** $3.0 \cdot 10^9$ does not fit in an int32, and the
  count wraps to a negative number of bits per weight on the first real model, having
  passed every small test. This is the classic bug of size accounting; the same one
  truncates files at 2 GB.

## Complexity

One pass over the weights, a handful of elementwise operations each, and no memory
beyond one matrix and its round trip. For Granite 350M that is under a second.

## In the wild

`mlx_lm.convert -q --q-bits 4 --q-group-size 64` applies exactly this function to
every `nn.Linear` and `nn.Embedding` and prints the bits per weight computed as
above. The measured outcome on this model is worth remembering: 8 bits is free (KL
0.003 nats), 4 bits is not (1.5 nats, and the greedy token changes at 3 of 12
positions), although the per-layer SQNR of about 20 dB looks unremarkable. Small
models have little redundancy; this is why mixed-precision recipes keep the embedding
and the first and last layers at 6 or 8 bits, and why the table of per-layer errors
is the starting point for choosing them. The next problem,
`mlx-quantized-checkpoint-load`, reads the file this scheme produces.
