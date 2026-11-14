## The idea

Uniform quantization spends its codes evenly between the smallest and the largest
value it has to cover. Weights are not spread evenly: most are small and a few are
large. If one grid covers a whole matrix, the few large weights set the step, and at 4
bits the step is so coarse that most weights land on two or three codes.

The fix is to make the grid local. Cut each row into short runs, and give every run its
own minimum (`bias`) and step (`scale`):

$$
\text{scale} = \frac{\max - \min}{2^{\text{bits}} - 1},
\qquad
\text{code} = \operatorname{round}\Big(\frac{w - \min}{\text{scale}}\Big),
\qquad
\hat w = \text{code} \cdot \text{scale} + \min .
$$

Inside a group this is ordinary affine quantization, in its simplest form: the minimum
maps to code 0 and the maximum to code $2^{\text{bits}} - 1$, both exactly, and every
other value is at most half a step from a grid point, because rounding to the nearest
integer moves a value by at most $0.5$ in units of `scale`. A group with a small range
gets a small step. That is the whole benefit.

The implementation is a question of shapes. An `(out, in)` matrix reshaped to
`(out, in // group_size, group_size)` has one group per position of the last axis,
because row-major order keeps consecutive elements of a row adjacent. A reduction over
the last axis then yields `(out, in // group_size)`, which is exactly the shape of
`scale` and `bias`.

## Reference solution (PyTorch)

```python
def quantize_groups(w, bits, group_size):
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.round((groups - lo[..., None]) / scale[..., None])
    codes = torch.clamp(codes, 0, levels).to(torch.uint8)
    return codes.reshape(out_dim, in_dim), scale, lo
```

The JAX and MLX versions are the same lines with `jnp.min` / `mx.min`, `clip`, and
`astype`.

Two details deserve a second look.

**The constant group.** When every element of a group is equal, `hi - lo` is 0, and
`(w - lo) / 0` is `0 / 0`, which is NaN. The `where` replaces the scale *before* the
division, so the division that is evaluated is `0 / 1`. Any positive scale would do,
since the codes are all 0 and the bias alone reproduces the value; the specification
picks 1. Constant groups are not exotic: pruned weights, padded embedding rows, and
freshly initialized adapter matrices are full of them.

**The clamp.** In exact arithmetic `(hi - lo) / scale` is exactly `levels`, so the
clamp looks redundant. In float32 the quotient can come out as `15.000001`, which still
rounds to 15, so it is redundant here in practice too. It is kept because the cast to
uint8 that follows wraps out-of-range values instead of saturating, and a clamp is the
cheap way to make that impossible.

## Common mistakes

- **Groups along the output axis.** Reducing over runs of rows within a column
  (`w.T.reshape(...)`, or a reduction over the wrong axis of the reshaped tensor)
  gives `scale` and `bias` of shape `(out / group_size, in)`. The kernel that later
  multiplies with the codes walks along a row and expects a new scale every
  `group_size` inputs, so the grouping has to follow the input axis.
- **Minimum and maximum over the whole row.** `w.amin(dim=-1)` instead of the group's
  own minimum collapses the scheme to per-row quantization. It is right when
  `group_size == in` and silently coarse otherwise: every group pays for the largest
  weight of the row.
- **Scale 0 on a constant group.** `scale = (hi - lo) / levels` without the special
  case produces `0 / 0`. Depending on the framework and the platform, the NaN is then
  cast to an arbitrary code, and a scale of 0 is stored either way.
- **Dividing by `2**bits`.** There are 16 codes at 4 bits and 15 steps between them.
  Dividing the range by 16 makes the step too small, so the maximum would need code
  16, which is clamped to 15, and the top of every group is off by a full step.
- **Truncating.** `astype(uint8)` on the unrounded quotient rounds toward zero. The
  error bound doubles from half a step to a full step, and the error is biased
  downward.
- **Hard-coding 15.** The same code has to serve 2, 3, 6, and 8 bits: mixed-precision
  recipes keep sensitive layers at 6 or 8 bits and the rest at 4.

## Complexity

One pass for the minimum and maximum and one for the codes: $O(\text{out} \cdot
\text{in})$ time. The output is `out * in` bytes of codes (before packing) plus two
floats per group.

## In the wild

**MLX.** `mx.quantize(w, group_size=64, bits=4)` is this problem with two refinements
and a packing step. Its documentation states the formula used here
($s = (\alpha - \beta) / (2^b - 1)$, $\hat w = \operatorname{round}((w - \beta) / s)$,
with scales and biases returned per group of consecutive elements in a row). The
implementation (MLX 0.32.4, `mlx/ops.cpp`) then adjusts the grid so that $0.0$ is
exactly representable: it anchors the grid at whichever end of the range is larger in
magnitude, gives the scale a sign so that the codes count from that end, and nudges
the scale so that the anchor is a whole number of steps away from zero. It also floors
the scale at $10^{-7}$, which is its answer to the constant group. The codes are then
packed into uint32 words, lowest bits first. The affine mode supports group sizes 32,
64, and 128, and 1 to 8 bits except 7. Scales and biases are stored in the dtype of
the weights, usually float16, so a 4-bit model with groups of 64 costs
$4 + 2 \cdot 16 / 64 = 4.5$ bits per weight.

**GGUF.** llama.cpp's legacy `Q4_1` format is the same idea with blocks of 32: one
float16 scale and one float16 minimum per block, and $w \approx d \cdot q + m$, for 5
bits per weight. `Q4_0` drops the minimum and uses a symmetric signed grid (4.5 bits
per weight). The K-quants add a second level: the per-block scales and minimums are
themselves quantized to 6 bits inside a superblock.

**What group-wise min-max does not do.** It minimizes nothing; it only guarantees the
half-step bound. Better quantizers keep this storage format and choose the numbers
more carefully: they clip the range when a slightly smaller range lowers the mean
squared error, or they choose codes to minimize the error of the layer's *output* on
calibration data (GPTQ, AWQ, and the learned quantizations shipped with `mlx-lm`). All
of them write the same triple of codes, scale, and bias.
