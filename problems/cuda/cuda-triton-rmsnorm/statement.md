Implement RMSNorm as a GPU kernel in Triton (`rmsnorm_kernel`) along with a PyTorch wrapper (`rms_norm`). Each Triton program normalizes one row of the input by stepping through its columns in fixed-size blocks and masking out-of-bounds positions in the final block.

## Background

A Triton kernel is a Python function compiled to run on a GPU. Launching `kernel[(n,)](args...)` starts `n` concurrent *programs* on a grid of size `n`. Every program runs the same code and queries its own index with `tl.program_id(0)`.

- Tensor arguments arrive as pointers to their first element. The kernel does not know their shapes or strides, and `ptr + k` advances the pointer by `k` elements.
- `tl.arange(0, BLOCK)` creates a 1D block of integers from `0` to `BLOCK - 1`. `BLOCK` must be a power of two passed by keyword as a compile-time constant (`tl.constexpr`). Arithmetic on blocks is elementwise.
- `tl.load(ptr + offsets, mask=m, other=v)` reads a block of elements from memory. Where `m` is `False`, no memory is read and the lane takes the value `v` (omitting `other` leaves it undefined).
- `tl.store(ptr + offsets, values, mask=m)` writes where `m` is `True` and converts `values` to the element type of `ptr`.
- This kernel also uses `tl.zeros((BLOCK,), tl.float32)`, `block.to(tl.float32)`, `tl.sum(block, axis=0)`, `tl.sqrt(...)`, and `for start in range(0, n, BLOCK):`. Python `int` and `float` launch arguments arrive as `int32` and `float32` scalars.

## Specification

```python
from learnai.kit import tritonshim as tl


@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr): ...


def rms_norm(x, weight, eps=1e-5, block=64): ...
```

### `rmsnorm_kernel`

Launch `rmsnorm_kernel` on a 1D grid of `(n_rows,)` programs, where `n_rows` is the number of rows and `n_cols` is the number of columns. `x_ptr` and `out_ptr` point to contiguous row-major buffers of `n_rows * n_cols` elements (element `[r, c]` is at offset `r * n_cols + c`), and `w_ptr` points to `n_cols` gain values.

Program `r = tl.program_id(0)` normalizes row `r`. For every column `c` from `0` to `n_cols - 1`, it computes:

$$
\text{out}_{r,c} = \frac{x_{r,c}}{\sqrt{\frac{1}{\text{n\_cols}} \sum_{j=0}^{\text{n\_cols}-1} x_{r,j}^2 + \varepsilon}} \; w_c
$$

where $\varepsilon$ is `eps` and $w_c$ is the gain at `w_ptr + c`.

- **Precision.** All three buffers share the same dtype (`float32` or `float16`). Compute the squares, their sum, the square root, and the division in `float32`, then let `tl.store` convert the final result back to the output dtype.
- **Blocking and masking.** Process each row in blocks of `BLOCK` columns. `n_cols` can be any positive integer, so it does not have to be a multiple of `BLOCK` and may even be smaller than `BLOCK`.
- **Memory safety.** Each program reads only from `x` and `w`, writes each of the `n_cols` elements of its own output row exactly once, and never reads or writes outside the three buffers or into another row's output.

![Two rows of five elements, read in blocks of four. The mask keeps each program's second block from touching the three elements past its row.](assets/tail-mask.svg)

### `rms_norm`

`rms_norm(x, weight, eps=1e-5, block=64)` applies `rmsnorm_kernel` along the last axis of `x`.

- `x` has shape `(..., d)` with one to four dimensions, and `weight` has shape `(d,)` with the same dtype as `x`. Either tensor may be a non-contiguous strided view (such as a slice or transpose).
- Launch `rmsnorm_kernel` **once**, with one program per row along the last axis and `BLOCK=block`.
- Return a new tensor with the same shape, dtype, and device as `x`. Do not modify `x` in place.

## Shapes

| name | shape | dtype |
|---|---|---|
| memory behind `x_ptr`, `out_ptr` | `n_rows * n_cols` elements, row-major | `float32` or `float16` |
| memory behind `w_ptr` | `n_cols` elements | same as `x_ptr` |
| `x` | `(..., d)`, 1 to 4 dimensions | `float32` or `float16` |
| `weight` | `(d,)` | same as `x` |
| result of `rms_norm` | same as `x` | same as `x` |

## Example

Consider two rows of five columns launched with `BLOCK=4` and `eps=1e-5`:

```python
x = [[3.0, 4.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
w = [1.0, 0.5, 1.0, 1.0, 2.0]

rmsnorm_kernel[(2,)](x, w, out, 5, 1e-5, BLOCK=4)
# out = [[1.341640, 0.894426, 0.0,      0.0,      0.0     ],
#        [0.999995, 0.499998, 0.999995, 0.999995, 1.999990]]
```

Program 0 processes row 0 in two blocks. The first block covers columns 0 through 3. The second block covers columns 4 through 7, where only column 4 is valid. Its mask is `[True, False, False, False]`, so the three masked-off lanes take `0.0` and the sum of squares is $9 + 16 = 25$. Dividing by `n_cols = 5` (not 8) gives a mean of 5, and the square root of $5 + 10^{-5}$ is $2.236070$. Dividing row 0 by $2.236070$ and scaling by `w` yields $[3, 4, 0, 0, 0] / 2.236070 \cdot w$.

Without the mask, program 0's second block would read `x[1, 0]`, `x[1, 1]`, and `x[1, 2]` into its sum, and program 1's second block would read three elements past the end of the tensor (Figure 1). Using `other=1.0` instead of `other=0.0` would add 3 to the sum of squares, producing $25 + 3 = 28$.

## Constraints

- $1 \le \text{n\_rows} \le 64$, $1 \le \text{n\_cols} \le 256$, and $10^{-6} \le \varepsilon \le 10^{-2}$. `BLOCK` is a power of two from 1 to 128.
- Float32 input entries have magnitude at most $10^4$. Float16 input entries have magnitude at most 2000 (whose square, $4 \times 10^6$, overflows `float16`). Gains are between 0.5 and 1.5.
- Outputs are compared against the formula evaluated in `float64` from the given input values. Tolerances are `rtol=1e-4`, `atol=1e-6` for `float32`, and `rtol=2e-3`, `atol=1e-3` for `float16`.
- Resource limits per launch: exactly `n_rows` programs, exactly `n_rows * n_cols` elements stored, and at most `3 * n_rows * n_cols` elements loaded (`x` across two passes plus `w` once).
- Use only the Triton primitives listed above, plus `tl.where`, `tl.rsqrt`, `tl.full`, comparisons, and arithmetic. Do not use Python lists, NumPy, or PyTorch inside the kernel.
- **CPU and GPU execution.** Standard test cases run on the CPU using `tritonshim`, which interprets the kernel in NumPy one program at a time and enforces strict checks that a real GPU omits. Unmasked out-of-bounds accesses raise `tl.OutOfBounds`, masked loads without `other=` produce NaN, concurrent writes to the same element raise `tl.DataRace`, and non-contiguous tensor arguments are rejected. Two hardware-marked cases run only on machines with an NVIDIA GPU, passing CUDA tensors and compiling the same kernel source with the real Triton compiler. (That GPU path and the shim's exact parity with Triton could not be tested during authoring, so the reference kernel follows Triton's official layer-norm tutorial construct by construct.)

## Why this matters

When a model needs an operation that standard library kernels do not fuse well, such as a normalization fused with neighboring layers or a custom quantized matmul, chaining PyTorch ops forces multiple passes over GPU memory. Triton solves this by compiling a single fused kernel, which is why `torch.compile` emits Triton and modern inference engines ship hand-written Triton kernels for normalization layers.

Most row-wise Triton kernels share this exact structure of one program per row, a loop over fixed-size blocks, and a masked tail. Both common bugs are silent on real GPUs. A missing mask does not crash. Instead, it folds elements from the next row into the current row's statistics or reads adjacent allocator memory, creating batch-dependent errors. Leaving the accumulator in `float16` overflows when squaring any value above 255, turning the norm infinite and zeroing out the largest activations.
