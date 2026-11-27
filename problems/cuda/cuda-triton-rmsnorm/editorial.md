## The idea

Launching `rmsnorm_kernel[(n_rows,)](...)` executes the kernel once per grid point concurrently. Inside the kernel, `tl.program_id(0)` identifies which program is running. Rather than operating on shaped tensors, the kernel manipulates pointers and fixed-size register blocks, using `tl.load` and `tl.store` to move blocks between registers and memory.

**One program per row.** Every element in a row shares the same root mean square, and no other row needs it. Assigning one program per row lets each program compute that statistic once and reuse it across all `n_cols` elements. One program per element would recompute the sum `n_cols` times, while assigning multiple rows per program would leave GPU parallelism unused.

**Fixed-size blocks and tail masking.** `tl.arange(0, BLOCK)` requires `BLOCK` to be a compile-time power of two, whereas `n_cols` can be any positive integer. For example, a row of 37 columns processed with `BLOCK=16` takes three blocks covering 48 columns, leaving 11 out-of-bounds lanes in the last block. For row 0, those 11 offsets point into the start of row 1. For the last row, they point past the end of the tensor. Guarding every memory access with `mask = cols < n_cols` prevents invalid reads and writes:

- On `tl.load`, masked-off lanes are not read and instead receive `other`. Using `other=0.0` is neutral for summation. Omitting `other` leaves those lanes undefined (typically stale register values).
- On `tl.store`, masked-off lanes are not written to memory.

**Two passes over the row.** Because a single block cannot hold an arbitrarily long row, and the root mean square must be known before any output is written, the kernel makes two passes. The first pass accumulates partial sums of squares in a `BLOCK`-element vector, then reduces it once with `tl.sum(acc, axis=0)` after the loop. The second pass reloads `x` and `w` in blocks and stores the normalized values.

**Internal `float32` arithmetic.** The largest finite `float16` value is 65504, so squaring any value with $|x| > 255$ overflows to infinity. Casting with `.to(tl.float32)` before squaring happens in registers at negligible cost, and `tl.store` converts the final output back to `float16` automatically. Divide the sum of squares by `n_cols` (the true column count, not the padded block count) and add `eps` inside the square root.

## Reference solution (PyTorch)

```python
@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * n_cols
    out_row = out_ptr + row * n_cols

    acc = tl.zeros((BLOCK,), tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rms = tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)

    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_row + cols, x / rms * w, mask=mask)


def rms_norm(x, weight, eps=1e-5, block=64):
    n_cols = x.shape[-1]
    rows = x.reshape(-1, n_cols).contiguous()
    out = torch.empty_like(rows)
    rmsnorm_kernel[(rows.shape[0],)](rows, weight.contiguous(), out, n_cols, eps, BLOCK=block)
    return out.reshape(x.shape)
```

Calling `.contiguous()` in the wrapper is essential. The kernel receives only base pointers and assumes row-major stride-1 layout (`row * n_cols + col`), whereas a strided view like `x[:, ::2]` has non-unit strides. When a tensor is already contiguous, `.contiguous()` returns it without copying. The output buffer uses `torch.empty_like` because the kernel writes every element.

On a GPU, `BLOCK` is a performance tuning knob. Larger blocks reduce loop iterations and widen vector operations per program. Triton's layer-norm tutorial selects the next power of two above `n_cols`, capped so the block fits in 64 KiB, and relies on tail masking for correctness at any size.

## Common mistakes

- **Omitting the mask.** Row 0 reads elements from row 1 into its sum and overwrites them on store, while the last row accesses memory past the end of the tensor. `tritonshim` raises `tl.OutOfBounds` or `tl.DataRace`, whereas a GPU fails silently.
- **Masking loads but not stores.** The statistics are correct, but each program overwrites the start of the next row and races with the program that owns it.
- **Using `cols <= n_cols`.** This includes one extra out-of-bounds element on every row.
- **Using `other=1.0` (or another non-zero value).** Every masked-off column adds 1 to the sum of squares. This bug passes whenever `n_cols` is a multiple of `BLOCK` and fails on non-multiple widths.
- **Dividing by the padded length.** The mean must be divided by `n_cols`, not the total number of lanes covered by the loop.
- **Adding `eps` outside the square root.** This looks almost identical on unit-scale activations but produces wrong outputs on small activations.
- **Skipping the `float32` cast.** Squaring `float16` values above 255 overflows to infinity and turns the normalized row into zeros.
- **Processing only one block without a loop.** Triton's tutorial uses a single block only because it sets `BLOCK >= n_cols`. With a fixed `BLOCK`, omitting the loop ignores all columns past `BLOCK`.
- **Launching one program per element.** This produces the right numbers but reads `x` `n_cols` times as often, exceeding the load limit.
- **Passing strided tensors or allocating the wrong output dtype.** Failing to call `.contiguous()` breaks strided views, and hardcoding `float32` for `out` breaks `float16` inputs.

## Complexity

The kernel launches `n_rows` programs. Each program runs $2 \lceil \text{n\_cols} / \text{BLOCK} \rceil$ loop iterations, loads $2 \cdot \text{n\_cols}$ elements of `x` and `n_cols` elements of `w`, and stores `n_cols` elements of `out`. It allocates no extra memory beyond the output tensor.

## In the wild

The forward pass of Triton's official layer-norm tutorial has this exact structure: a row pointer offset by `row * stride`, a loop `for off in range(0, N, BLOCK_SIZE)`, masked loads with `other=0.0`, and a `float32` accumulator reduced with `tl.sum` (plus mean subtraction). Kernels emitted by `torch.compile` for fused normalizations follow the same pattern.

**Verification note.** All judge checks were verified on CPU under `tritonshim`, which was written from Triton's documentation without a local Triton installation for side-by-side testing. Three shim details rest on reading the docs rather than explicit guarantees: Python scalars arrive as `int32`/`float32`, integer `/` integer yields `float32`, and block shapes must be powers of two. The reference kernel does not rely on integer `/` integer, and the official tutorial relies on the other two in the same way. (The tutorial also passes a list to `tl.zeros([BLOCK_SIZE], dtype=tl.float32)` where the API docs specify a tuple; the shim accepts both.) The GPU test cases that invoke the real Triton compiler have not been run.
