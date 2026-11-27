## The idea

Group-wise 4-bit quantization (`groupwise-4bit`) stores two floats alongside every group of codes. With 32 weights per group and float16 parameters, that costs $4 + 32 / 32 = 5$ bits per weight, meaning a fifth of the file is metadata. K-quants compress the metadata with quantization as well. Because the 8 block scales in a superblock are non-negative numbers of similar magnitude, they can be stored as 6-bit multiples of one shared float16 unit, and the same applies to the 8 block minimums. Across 256 weights, the bit budget is:

$$
\underbrace{256 \cdot 4}_{\text{codes}} + \underbrace{8 \cdot 6 + 8 \cdot 6}_{\text{block fields}} + \underbrace{2 \cdot 16}_{\text{units}} = 1152 \text{ bits} = 4.5 \text{ bits per weight}.
$$

**The sign convention.** The decoder evaluates $\hat w = D \cdot q - M$ with unsigned integer $q$, $D \ge 0$, and $M \ge 0$. A block's grid is therefore $\{-M, -M + D, \dots, -M + 15 D\}$, so its lowest point $-M$ can never be positive. If all weights in a block are positive, the closest valid grid start is $0$. Clamping $lo_b$ to at most $0$ in step 1 enforces this constraint. It reduces resolution for all-positive blocks because their 16 levels must cover $[0, \max]$ instead of $[\min, \max]$, but real neural network weight blocks almost always straddle zero, and in return the offset needs no sign bit.

**Rounding in the right order.** Step 1 gives each block an ideal grid. Steps 2 and 3 quantize that grid by rounding `d` and `dmin` to float16 and rounding the per-block scales and minimums to 6-bit integers. If you computed the 4-bit codes on the ideal grid from step 1, the decoder would still reconstruct the quantized grid, breaking the half-step optimality guarantee. Computing the codes last, using the exact float32 products of the float16 units and 6-bit integers, maps every weight to the nearest point on the grid the decoder will actually use.

**The error bound.** Any weight inside the reconstructed grid lies within $D_b / 2$ of a grid point. A weight can also fall slightly outside the reconstructed grid because 6-bit rounding moves the bottom endpoint by at most $d_{\min} / 2$ and the top endpoint by at most $d_{\min} / 2 + 15 \cdot d / 2$. Combining these gives $|w - \hat w| \le \tfrac12 D_b + \tfrac{15}{2} d + \tfrac12 d_{\min}$, up to the float16 rounding of `d` and `dmin` (about one part in 2048). The $\tfrac{15}{2} d$ term shows why mixing blocks of very different ranges in one superblock hurts accuracy. The largest block sets $d$, and any block whose ideal scale falls below $d / 2$ rounds to `scales = 0` and collapses to a single constant value.

## Reference solution (PyTorch)

```python
def _six_bit(value, largest):
    safe = torch.where(largest > 0, largest, torch.ones_like(largest))
    return torch.where(largest > 0, torch.round(63 * value / safe), 0.0)


def q4k_encode(w):
    n = w.shape[0]
    blocks = w.reshape(n, 8, 32)
    lo = torch.clamp(blocks.amin(dim=-1), max=0.0)
    hi = blocks.amax(dim=-1)
    block_scale = (hi - lo) / 15
    block_min = -lo
    max_scale = block_scale.amax(dim=-1, keepdim=True)
    max_min = block_min.amax(dim=-1, keepdim=True)
    scales = _six_bit(block_scale, max_scale)
    mins = _six_bit(block_min, max_min)
    d = (max_scale / 63).to(torch.float16)
    dmin = (max_min / 63).to(torch.float16)
    step = d.to(torch.float32) * scales
    offset = dmin.to(torch.float32) * mins
    safe = torch.where(step > 0, step, torch.ones_like(step))
    codes = torch.clamp(torch.round((blocks + offset[..., None]) / safe[..., None]), 0, 15)
    codes = torch.where(step[..., None] > 0, codes, 0.0)
    return (
        codes.reshape(n, 256).to(torch.uint8),
        scales.to(torch.uint8),
        mins.to(torch.uint8),
        d[:, 0],
        dmin[:, 0],
    )


def q4k_decode(codes, scales, mins, d, dmin):
    n = codes.shape[0]
    step = d.to(torch.float32)[:, None] * scales.to(torch.float32)
    offset = dmin.to(torch.float32)[:, None] * mins.to(torch.float32)
    q = codes.to(torch.float32).reshape(n, 8, 32)
    return (step[..., None] * q - offset[..., None]).reshape(n, 256)
```

The JAX and MLX implementations are line-for-line translations. All three frameworks round float32 to the nearest float16 identically, including subnormal values reached by small weights ($d < 6.1 \cdot 10^{-5}$).

## Common mistakes

- **Allowing a positive block minimum.** Without clamping $lo_b$ to at most $0$, an all-positive block produces $\text{min}_b < 0$. Casting a negative value to `uint8` wraps or saturates depending on the framework.
- **Computing codes on the ideal grid.** Using step 1's `block_scale` and `block_min` instead of the reconstructed grid leaves every weight slightly off after decoding, throwing away precision.
- **Using float32 units in step 4.** Computing `step` and `offset` before casting `d` and `dmin` to float16 introduces up to a 1-in-2048 discrepancy in the step size, which over 15 steps is enough to flip codes near a rounding boundary.
- **Flooring the 6-bit fields.** Truncating instead of rounding biases every block's step and offset downward, shrinking the grid and clipping the largest weights.
- **Taking the unit from the first block.** Dividing by `block_scale[:, 0]` instead of `max_scale` causes larger blocks to exceed 63.
- **Dividing by 64 or 16 instead of 63 or 15.** A 6-bit unsigned integer has 64 levels with maximum value 63. A 4-bit unsigned code has 16 levels spanning 15 intervals.
- **Missing zero guards.** A superblock with no negative weights has $\text{max\_min} = 0$, and an all-zero superblock has both maxima equal to $0$. Guard divisions with `where` so these fields become $0$ instead of `NaN`.
- **Decoding in `uint8`.** Multiplying `scales * codes` before casting to float32 overflows `uint8` at 256, while the maximum product is $63 \cdot 15 = 945$.
- **Adding the offset instead of subtracting it.** The stored minimum is the negated lower bound, so `q4k_decode` must subtract $d_{\min} \cdot \text{mins}_b$.
- **Truncating the codes or returning float32 units.** Codes must be rounded to the nearest integer, and `d` and `dmin` must be returned as float16.

## Complexity

The encoder makes one pass over the weights to find block extremes and a second pass to compute codes, taking $O(S \cdot 256)$ time. All intermediate parameter work operates on `(S, 8)` tensors.

## In the wild

In **llama.cpp** (`ggml/src/ggml-quants.c`), the struct `block_q4_K` stores two float16 fields `d` and `dmin`, 12 bytes of packed 6-bit `scales` and minimums, and 128 bytes of packed 4-bit `qs`. The decoder `dequantize_row_q4_K` computes `d1 = d * sc`, `m1 = min * m`, and `d1 * (q[l] & 0xF) - m1`, which matches `q4k_decode`. The reference quantizer `quantize_row_q4_K_ref` computes `inv_scale = 63.f / max_scale` and `ls = nearest_int(inv_scale * scales[j])` clamped to 63, stores `GGML_FP32_TO_FP16(max_scale / 63.f)` and `GGML_FP32_TO_FP16(max_min / 63.f)`, and recomputes every code as `nearest_int((x + dm) / d)` clamped to $[0, 15]$ using the reconstructed `d` and `dm`. In `make_qkx2_quants`, `if (min > 0) min = 0;` is the clamp from step 1, and the minimum is returned negated (`*the_min = -min`).

**What differs in llama.cpp:**

- *Per-block parameter search.* Instead of plain min-max, llama.cpp weights each element (by magnitude in the default path, or by importance-matrix statistics in the `imatrix` path) and searches candidate scales using weighted least squares. Step 1 here is the simplest procedure with the same interface.
- *Bit packing.* The 16 six-bit fields are packed into 12 bytes (`get_scale_min_k4`). For blocks 0 to 3, the low 6 bits of byte `j` hold the scale and byte `j + 4` holds the minimum. For blocks 4 to 7, byte `j + 4` holds the low 4 bits of both fields, while their top 2 bits sit in the unused upper bits of bytes `j - 4` and `j`. Codes are packed two blocks at a time, with byte `l` of each 64-weight pair holding weight `l` in the low nibble and weight `l + 32` in the high nibble.
- *Zero-step blocks.* When a block's reconstructed step is $0$, llama.cpp skips re-rounding and keeps the search codes, whereas this problem sets them to $0$. Both decode to the same constant value.

**The K-quant family.** `Q5_K` uses the same superblock structure with 5-bit codes, while `Q6_K`, `Q3_K`, and `Q2_K` use 16 blocks of 16 weights (`Q6_K` uses 8-bit block scales and no minimum). The `_S`, `_M`, and `_L` suffixes in GGUF filenames are mixed-precision recipes that assign different block formats to different tensors, as covered in `bit-allocation`.
