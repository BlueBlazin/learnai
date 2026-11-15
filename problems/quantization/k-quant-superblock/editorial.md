## The idea

Group-wise 4-bit quantization (`groupwise-4bit`) stores, for every group, two floats
next to the codes. With groups of 32 and float16 parameters that is
$4 + 32 / 32 = 5$ bits per weight: a fifth of the file is metadata. K-quants attack
the metadata with the same tool used on the weights: quantize it. The eight block
scales of a superblock are positive numbers of similar size, so they are stored as
6-bit multiples of one shared float16 unit, and likewise the eight minimums. Per
superblock:

$$
\underbrace{256 \cdot 4}_{\text{codes}} + \underbrace{8 \cdot 6 + 8 \cdot 6}_{\text{block fields}} + \underbrace{2 \cdot 16}_{\text{units}} = 1152 \text{ bits} = 4.5 \text{ bits per weight}.
$$

**The sign convention.** The decoder computes $\hat w = D \cdot q - M$ with unsigned
$q$, $D \ge 0$, and $M \ge 0$. The grid of a block is therefore
$\{-M, -M + D, \dots, -M + 15 D\}$, and its bottom, $-M$, cannot be positive. A block
whose weights are all positive cannot have its own minimum as the bottom of the
grid; the best the format allows is 0. That is the clamp of step 1, and it costs
such a block some resolution: its 16 levels cover $[0, \max]$ instead of
$[\min, \max]$. Real weight blocks almost always straddle zero, so the format loses
little, and in exchange the offset needs no sign bit.

**Two roundings, in the right order.** After step 1 every block has an ideal grid.
Step 3 moves each grid: the step $D_b$ becomes the nearest multiple of $d$ and the
offset $M_b$ the nearest multiple of $d_{\min}$, and the units themselves were
rounded to float16. If the codes had been computed on the ideal grid, each would
now be decoded on a different one, and the half-step guarantee would be gone. So
the codes are computed **last**, against exactly the numbers the decoder will
compute: float16 unit times 6-bit integer, in float32. Rounding a weight to the
nearest point of the grid that will actually be used is optimal for that grid,
whatever happened upstream.

**The error bound.** Inside the stored grid, a weight is at most $D_b / 2$ from a
grid point. A weight can also fall outside it, because the stored grid's ends moved:
the offset by at most $d_{\min} / 2$, and the top by that plus 15 times the step's
rounding error, $15 \cdot d / 2$. Hence
$|w - \hat w| \le \tfrac12 D_b + \tfrac{15}{2} d + \tfrac12 d_{\min}$, up to the
float16 rounding of the units (one part in 2048). The second term is why a
superblock should not mix blocks of very different range: $d$ is set by the largest
block, and a small block's step is rounded to a multiple of it. A block whose scale
is below $d / 2$ gets `scales = 0` and collapses to a single value.

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

The JAX and MLX versions are line-for-line translations; all three frameworks round
a float32 to the nearest float16 identically, including in the subnormal range that
small weights reach ($d$ below $6.1 \cdot 10^{-5}$).

## Common mistakes

- **A positive block minimum.** Without the clamp, a block of positive weights has
  $\text{min}_b < 0$, and the "6-bit unsigned" field receives a negative number,
  which wraps or saturates depending on the framework.
- **Codes on the ideal grid.** Skipping the second pass is the classic error. The
  result decodes almost right, off by a fraction of a step everywhere, which is
  exactly the precision the format was supposed to keep.
- **Float32 units in the second pass.** Computing the codes before the float16
  cast is the same mistake at a smaller scale: the step differs by up to one part
  in 2048, which over 15 steps is enough to change codes near a tie.
- **Flooring the 6-bit fields.** Truncation biases every block's step and offset
  downward; the top of every grid falls short and the largest weights are clipped.
- **The unit from the first block.** The unit must come from the *largest* scale
  (and minimum) of the superblock, or larger blocks need a field above 63.
- **63 versus 64, 15 versus 16.** A 6-bit field has 64 values and its largest is
  63; a 4-bit code has 16 values and 15 steps.
- **No guard for zero.** A superblock without negative weights has
  $\text{max\_min} = 0$, and an all-zero superblock has both maxima 0; the fields
  are then 0 by definition, not $0/0$.
- **Decoding in uint8.** `scales * codes` on byte tensors wraps at 256; the largest
  product is 945.
- **Adding the offset.** The stored minimum is the *negated* block minimum, and the
  decoder subtracts it.
- **Truncating the codes**, or returning `d` and `dmin` as float32.

## Complexity

One pass over the weights for the block extremes and one for the codes:
$O(S \cdot 256)$ time, with all the parameter work on `(S, 8)` tensors.

## In the wild

**llama.cpp** (`ggml/src/ggml-quants.c`). The block is
`block_q4_K`: two float16 fields `d` and `dmin`, 12 bytes of `scales`, and 128 bytes
of `qs`. `dequantize_row_q4_K` computes `d1 = d * sc`, `m1 = min * m`, and
`d1 * (q[l] & 0xF) - m1`, which is `q4k_decode`. The reference quantizer
`quantize_row_q4_K_ref` computes `inv_scale = 63.f/max_scale` and
`ls = nearest_int(inv_scale*scales[j])` clamped to 63, stores
`GGML_FP32_TO_FP16(max_scale/63.f)` and `GGML_FP32_TO_FP16(max_min/63.f)`, and then
recomputes every code as `nearest_int((x + dm)/d)` clamped to $[0, 15]$ with `d` and
`dm` rebuilt from the stored fields: steps 2 to 4 of this problem. In
`make_qkx2_quants`, the per-block search, `if (min > 0) min = 0;` is the clamp of
step 1, and the block minimum is returned negated (`*the_min = -min`).

**What differs.**

- *The per-block search.* llama.cpp does not use min-max for a block. It weights
  each element (by its magnitude in the reference path, by importance-matrix data in
  the `imatrix` path) and tries a range of candidate scales, solving a small
  least-squares problem for each, to minimize the weighted error. This problem's
  step 1 is the simplest procedure with the same output type.
- *Bit packing.* The 16 six-bit fields occupy 12 bytes: for blocks 0 to 3, the low
  6 bits of byte `j` hold the scale and of byte `j + 4` the minimum; for blocks 4 to
  7, byte `j + 4` holds the low 4 bits of the scale and of the minimum, and their
  top 2 bits go into the two spare bits of bytes `j - 4` and `j`
  (`get_scale_min_k4`). The codes are packed two blocks at a time: byte `l` of a
  64-weight half-block pair holds weight `l` in its low nibble and weight `l + 32`
  in its high nibble.
- *A block with step 0.* llama.cpp skips the re-rounding for such a block and keeps
  the codes of its search; here they are defined as 0. The decoded values are the
  same, since the step is 0.

**The family.** `Q5_K` uses the same superblock with 5-bit codes; `Q6_K` uses 16
blocks of 16 with 8-bit block scales and no minimum; `Q3_K` and `Q2_K` also use 16
blocks of 16. The `_S`, `_M`, and `_L` suffixes of GGUF file names do
not name block formats: they are recipes that assign different block formats to
different tensors of a model, the problem of `bit-allocation`.
