## The idea

Uniform quantization places codes evenly between the minimum and maximum values of a range. Neural network weights are rarely spread evenly, because most values are small while a few are large. When one grid covers an entire matrix or row, those few outliers set a coarse step size, leaving most 4-bit weights collapsed onto two or three codes.

Group-wise quantization solves this by making the grid local. Cutting each row into short consecutive runs gives every group its own minimum (`bias`) and step size (`scale`):

$$
\text{scale} = \frac{\max - \min}{2^{\text{bits}} - 1},
\qquad
\text{code} = \operatorname{round}\!\Big(\frac{w - \min}{\text{scale}}\Big),
\qquad
\hat w = \text{code} \cdot \text{scale} + \min.
$$

Inside each group, this is standard affine quantization. The minimum maps to code $0$ and the maximum maps to code $2^{\text{bits}} - 1$ with zero error. Every other element lands within half a step ($\text{scale} / 2$) of its original value because rounding moves a scaled value by at most $0.5$. Groups with a narrow range receive a small step size without being distorted by outliers elsewhere in the row.

To implement this without loops, reshape `w` from `(out, in)` to `(out, in // group_size, group_size)`. Row-major order keeps consecutive elements of a row adjacent, so the last axis holds one group. Reducing over the last axis produces `(out, in // group_size)`, which already matches the required shape of `scale` and `bias`.

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

The JAX and MLX implementations use the same structure with `jnp.min` / `mx.min`, `clip`, and `astype`.

**Handling constant groups.** When all elements in a group are equal, `hi - lo` is `0`, so dividing `(groups - lo)` by `0` produces `NaN`. Using `where` replaces zero scales with `1` *before* the division, so the quotient evaluates to `0 / 1 = 0`. Any positive scale would recover the value because all codes are `0` and `bias` alone reconstructs the group. The specification fixes this scale to `1`. Constant groups appear regularly in pruned weights, padded embeddings, and zero-initialized adapters.

**Clamping before casting.** In exact arithmetic, `(hi - lo) / scale` equals `levels`, and in float32 a slight overshoot like `15.000001` still rounds to `15`. Keeping `clamp` before casting to `uint8` is cheap insurance, because unsigned integer casts wrap out-of-range values instead of saturating.

## Common mistakes

- **Grouping along the output axis.** Reducing across runs of rows within a column (`w.T.reshape(...)` or reducing the wrong reshaped axis) produces `scale` and `bias` of shape `(out / group_size, in)`. The matmul kernel walks along each row and expects a new scale every `group_size` inputs, so groups must follow the input axis.
- **Computing min and max over the whole row.** Using `w.amin(dim=-1)` collapses the scheme to per-row quantization. It passes when `group_size == in` but silently produces overly coarse grids otherwise, forcing every group to pay for the largest outlier in the row.
- **Leaving scale as 0 on a constant group.** Computing `scale = (hi - lo) / levels` without guarding against `hi == lo` creates `0 / 0 = NaN`. Casting that `NaN` to `uint8` produces platform-dependent garbage codes alongside a zero scale.
- **Dividing by `2**bits`.** A 4-bit unsigned integer has 16 codes (`0` through `15`) and 15 intervals between them. Dividing by `16` makes the step too small, pushing the maximum toward code `16` (which clamps to `15`) and leaving the top of every group off by a full step.
- **Truncating instead of rounding.** Casting `(groups - lo) / scale` directly to `uint8` truncates toward zero. This doubles the maximum error from half a step to a full step and introduces a downward bias.
- **Hard-coding 15.** Quantization must support any `bits` from 2 to 8, since mixed-precision recipes often keep sensitive layers at 6 or 8 bits while quantizing the rest to 4 bits.

## Complexity

Computing group extrema takes one pass over `w`, and quantizing the elements takes a second pass, for $O(\text{out} \cdot \text{in})$ total time. Storage before bit-packing is `out * in` bytes for `codes` plus two float32 values per group for `scale` and `bias`.

## In the wild

**MLX.** `mx.quantize(w, group_size=64, bits=4)` implements this layout with two refinements and a bit-packing step. Its documentation states the same affine min-max formula per group along each row. In MLX 0.32.4 (`mlx/ops.cpp`), the implementation adjusts the grid so `0.0` is exactly representable by anchoring at the larger-magnitude endpoint, signing the scale so codes count from that anchor, and nudging the scale so zero falls on an integer step. It also floors the scale at $10^{-7}$ to handle constant groups, then packs codes into `uint32` words lowest bits first. Affine mode supports group sizes 32, 64, and 128, and 1 to 8 bits except 7. Storing scales and biases in `float16` for 4-bit weights with group size 64 yields $4 + 2 \cdot 16 / 64 = 4.5$ bits per weight.

**GGUF.** `llama.cpp`'s `Q4_1` format applies the same design to blocks of 32 weights, storing one float16 scale and one float16 minimum per block ($w \approx \text{scale} \cdot \text{code} + \min$) for 5 bits per weight. `Q4_0` drops the minimum and uses a symmetric signed grid for 4.5 bits per weight. K-quants add a second level of quantization by quantizing the per-block scales and minimums to 6 bits inside larger superblocks.

**Beyond min-max.** Group-wise min-max quantization guarantees the half-step bound but does not minimize reconstruction error. Advanced quantizers such as GPTQ, AWQ, and learned recipes in `mlx-lm` keep the exact same `(codes, scale, bias)` storage format, while clipping ranges to lower mean squared error or choosing codes to minimize layer output error on calibration data.
