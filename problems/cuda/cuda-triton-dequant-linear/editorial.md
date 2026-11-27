## The idea

Each dequantized weight is used for a single multiplication. By unpacking and scaling the weights in registers inside the dot-product loop, the kernel reads only the packed words, scales, biases, and activations, and writes only the final output scalar.

**One program per output element.** Each output $y_{b,o}$ is the dot product of row $b$ of `x` with row $o$ of $\hat W$, independent of all other outputs. Launching on a 2D grid of `(n_out, batch)` lets every program run the same blocked loop over `n_in` columns as `cuda-triton-rmsnorm`, with a `float32` accumulator and a masked tail.

**Three indices per column.** For column $c$ of output row $o$, three expressions locate the data:

| quantity | expression | reason |
|---|---|---|
| packed word | `words_row + c // 8` | eight 4-bit codes per `uint32` |
| 4-bit code | `(word >> 4 * (c % 8)) & 0xF` | code 0 sits in the lowest bits |
| scale and bias | `scales_row + c // group_size` | one scale/bias pair per group |

All three expressions evaluate across an entire block of columns at once (`cols // 8`, `cols % 8`, `cols // group_size`). A `tl.load` with a block of computed offsets fetches one value per lane even when multiple lanes point to the same address. Unlike `cuda-triton-rmsnorm`, where loads were strictly contiguous, these gathered offsets decouple `BLOCK` from `group_size`. A block of 16 columns with `group_size=24` switches groups mid-block, and a block of 64 with `group_size=8` spans eight groups, with no special-case branching required.

**Unsigned right shifts.** Loading `words` as `tl.uint32` keeps the shift unsigned, so `>>` always shifts in zeros. Casting the shift amount to `tl.uint32` makes the unsigned operation explicit (Triton promotes mixed signed/unsigned operands of equal width to unsigned anyway). Masking with `& 0xF` after shifting is also safer than the signed two-shift bit-field pattern `(word << a) >> b`, which fails under `tritonshim` whenever the intermediate value is negative.

**`float32` accumulation and safe masking.** Cast the unpacked code, scale, and bias to `float32` before computing the weight and multiplying by `x`. A `float16` accumulator both rounds each partial sum to 11 mantissa bits and overflows 65504 when summing a few hundred large products. Loading `x` with `other=0.0` ensures masked-off columns contribute zero, and providing `other=` on the other three loads prevents multiplying by undefined values.

## Reference solution (PyTorch)

```python
@tl.jit
def qlinear_kernel(
    x_ptr, words_ptr, scales_ptr, biases_ptr, y_ptr, n_in, n_out, group_size, BLOCK: tl.constexpr
):
    out_row = tl.program_id(0)
    batch_row = tl.program_id(1)
    x_row = x_ptr + batch_row * n_in
    words_row = words_ptr + out_row * (n_in // 8)
    scales_row = scales_ptr + out_row * (n_in // group_size)
    biases_row = biases_ptr + out_row * (n_in // group_size)

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_in, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_in
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        word = tl.load(words_row + cols // 8, mask=mask, other=0)
        code = (word >> ((cols % 8) * 4).to(tl.uint32)) & 0xF
        group = cols // group_size
        scale = tl.load(scales_row + group, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(biases_row + group, mask=mask, other=0.0).to(tl.float32)
        acc += x * (code.to(tl.float32) * scale + bias)
    tl.store(y_ptr + batch_row * n_out + out_row, tl.sum(acc, axis=0))


def qlinear(x, words, scales, biases, group_size, block=64):
    n_out, n_groups = scales.shape
    n_in = n_groups * group_size
    rows = x.reshape(-1, n_in).contiguous()
    y = torch.empty((rows.shape[0], n_out), dtype=x.dtype, device=x.device)
    grid = (n_out, rows.shape[0])
    qlinear_kernel[grid](
        rows,
        words.contiguous(),
        scales.contiguous(),
        biases.contiguous(),
        y,
        n_in,
        n_out,
        group_size,
        BLOCK=block,
    )
    return y.reshape(*x.shape[:-1], n_out)
```

This kernel prioritizes clarity and independence from `group_size`. It loads each packed word 8 times and each scale/bias `group_size` times. Those repeated loads hit the same cache line, though they still issue redundant load instructions. Two standard optimizations cut load traffic at the cost of tying `BLOCK` to the quantization layout:

- Load `BLOCK // 8` words once and unpack all eight nibbles via a 2D broadcast (`word[:, None] >> shifts[None, :]`).
- Process one block per group using the identity from `quantized-matmul`, $\sum_{c \in g} x_c (q_c s_g + b_g) = s_g \sum x_c q_c + b_g \sum x_c$, loading `scale` and `bias` as scalars once per group.

## Common mistakes

- **Unpacking nibbles from high bits to low (`4 * (7 - c % 8)`).** This reverses every block of eight weights.
- **Loading one scale and bias per block.** Using the first column's group for the entire block works only when `BLOCK` divides `group_size` evenly and breaks as soon as `BLOCK` changes.
- **Omitting the row offset on `scales` or `biases`.** Every output row ends up using row 0's parameters, which passes only on single-row tests.
- **Performing a signed right shift.** An arithmetic shift turns codes 8 through 15 into $-8$ through $-1$. `tritonshim` catches this immediately by rejecting shifts on negative signed values.
- **Placing the bias outside the multiplication (`x * code * scale + bias`).** The bias is part of the weight and must be multiplied by `x`. Leaving it outside adds $\text{group\_size} \cdot b$ per group instead of $b \sum x$.
- **Accumulating in `float16`.** This matches to roughly three digits on short rows but drifts or overflows on longer rows and larger activations.
- **Omitting masks or `other=`.** Missing masks read into adjacent rows or past the end of the tensor, and missing `other=` multiplies undefined values. Both fail silently on a real GPU.
- **Transposing the output index or grid axes.** Storing at `o * batch + b` or launching on `(batch, n_out)` writes a transposed result.
- **Dequantizing in the wrapper and calling `matmul`.** This produces the right numbers but violates the fused-kernel requirement by materializing the float weight matrix in memory.
- **Passing strided tensors or wrong output dtype.** Skipping `.contiguous()` breaks strided inputs, and allocating `y` as `float32` breaks `float16` callers.

## Complexity

The kernel launches `n_out * batch` programs. Each program runs $\lceil \text{n\_in} / \text{BLOCK} \rceil$ loop iterations, loads `4 * n_in` elements (one activation, word, scale, and bias per column), and stores 1 output element. It uses no extra memory beyond the output tensor.

## In the wild

MLX's `quantized_matmul` performs this exact fused operation for the same 4-bit layout. Its Metal counterpart for standalone dequantization appears in `mlx-metal-kernel-dequant`, which assigns one thread per weight rather than one program per output. GPTQ formats pack 4-bit codes into `int32` words (which is why signed-shift bugs are common in practice) and use a zero point per group instead of an additive bias, while GGUF's `Q4_K` packs blocks of 32 codes with 6-bit scales. All of these share the same loop structure and differ mainly in the three index expressions. Production kernels also tile across the batch dimension so that each packed word is unpacked once and reused across several batch vectors.

**Verification note.** All judge checks were verified on CPU under `tritonshim`. The shim implements Triton's documented type promotion rules (unsigned wins over signed of the same width, Python integers adopt the block's dtype, and integer `//` and `%` truncate toward zero). Without explicit Triton documentation on two points, the shim assumes `>>` is a logical shift on unsigned integers (refusing negative signed shifts) and that PyTorch `uint32` tensors pass as `uint32` pointers. The reference kernel never shifts a signed value, so it depends only on the unsigned shift behavior. The GPU test cases that invoke the real Triton compiler have not been run.
