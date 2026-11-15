Write RMSNorm as a GPU kernel in Triton, the language PyTorch's own compiler generates
its CUDA kernels in. One program normalizes one row, reading it in fixed-size blocks.
The formula is the one from `rmsnorm`; the work is in the last block of each row,
which hangs over the end of the row and must neither read nor write what is there.

## Specification

```python
from whetstone.kit import tritonshim as tl


@tl.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_cols, eps, BLOCK: tl.constexpr): ...


def rms_norm(x, weight, eps=1e-5, block=64): ...
```

**The language.** A Triton kernel is a Python function that is compiled, not run. It
is launched on a *grid*: `kernel[(n,)](args...)` starts `n` *programs*, all running
the same code concurrently, each told which one it is by `tl.program_id(0)`.

- A tensor argument arrives as a **pointer** to its first element; the kernel knows
  neither its shape nor its strides. `ptr + k` points `k` elements further.
- `tl.arange(0, BLOCK)` is a block of the integers `0` to `BLOCK - 1`; `BLOCK` must
  be a compile-time constant (a parameter annotated `tl.constexpr`, passed by
  keyword) and a power of two. Arithmetic on blocks is elementwise.
- `tl.load(ptr + offsets, mask=m, other=v)` reads a block of elements. Where `m` is
  false nothing is read and the value is `v`; without `other` it is undefined.
  `tl.store(ptr + offsets, values, mask=m)` writes where `m` is true and converts
  `values` to the element type behind the pointer.
- `tl.zeros((BLOCK,), tl.float32)`, `block.to(tl.float32)`, `tl.sum(block, axis=0)`,
  `tl.sqrt(...)`, and `for start in range(0, n_cols, BLOCK):` are what else this
  kernel needs. Python integers and floats passed at launch (`n_cols`, `eps`) arrive
  as int32 and float32 scalars.

**`rmsnorm_kernel`** is launched on a grid of `(n_rows,)`. `x_ptr` and `out_ptr`
point to `n_rows * n_cols` elements in row-major order (element `[r, c]` is at offset
`r * n_cols + c`), and `w_ptr` to `n_cols` gains. Program `r` computes, for every
column `c` of its row,

$$
\text{out}_{r,c} = \frac{x_{r,c}}{\sqrt{\frac{1}{\text{n\_cols}} \sum_{j=0}^{\text{n\_cols}-1} x_{r,j}^2 + \varepsilon}} \; w_c .
$$

- The three tensors have the same element type, **float32 or float16**. The squares,
  their sum, the root, and the quotient are computed in **float32**, and the result
  is converted once, by the store.
- The row is processed in blocks of `BLOCK` columns. `n_cols` is any positive
  integer: it need not be a multiple of `BLOCK`, and may be smaller.
- The kernel reads `x` and the gains, and writes each of the `n_cols` elements of its
  own output row exactly once. It touches no element outside the three tensors and
  no element of another row's output.

![The worked example in memory: two rows of five elements, read in blocks of four. The second block of each program covers three elements that are not part of its row; the mask keeps them from being read or written.](assets/tail-mask.svg)

**`rms_norm(x, weight, eps, block)`** applies the kernel to a tensor.

- `x`: float32 or float16, shape `(..., d)` with one to four dimensions; `weight`:
  shape `(d,)`, the dtype of `x`. Either may be a strided view (every second column
  of a larger tensor, a transposed batch).
- It launches `rmsnorm_kernel` **once**, with one program per vector along the last
  axis and `BLOCK=block`, and returns a new tensor with the shape, dtype, and device
  of `x`. `x` is not modified.

## Shapes

| name | shape | dtype |
|---|---|---|
| memory behind `x_ptr`, `out_ptr` | `n_rows * n_cols` elements, row-major | float32 or float16 |
| memory behind `w_ptr` | `n_cols` elements | the same |
| `x` | `(..., d)`, one to four dimensions | float32 or float16 |
| `weight` | `(d,)` | the dtype of `x` |
| result of `rms_norm` | the shape of `x` | the dtype of `x` |

## Example

Two rows of five columns, launched with `BLOCK=4` and `eps = 1e-5`:

```python
x = [[3.0, 4.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
w = [1.0, 0.5, 1.0, 1.0, 2.0]

rmsnorm_kernel[(2,)](x, w, out, 5, 1e-5, BLOCK=4)
# out = [[1.341640, 0.894426, 0.0,      0.0,      0.0     ],
#        [0.999995, 0.499998, 0.999995, 0.999995, 1.999990]]
```

Program 0 reads its row in two blocks. The first covers columns 0 to 3. The second
covers columns 4 to 7, of which only column 4 exists in the row: the mask is
`[True, False, False, False]`, the three masked values count as 0, and the sum of
squares is $9 + 16 = 25$. The mean is $25 / 5 = 5$ (five columns, not eight), the
root of $5 + 10^{-5}$ is $2.236070$, and the row becomes
$[3, 4, 0, 0, 0] / 2.236070 \cdot w$.

Without the mask, program 0's second block would read `x[1, 0]`, `x[1, 1]`, and
`x[1, 2]` into its sum, and program 1's second block would read three elements past
the end of the tensor (Figure 1). Loading the masked elements as 1 instead of 0
would make the sum $25 + 3$.

## Constraints

- $1 \le \text{n\_rows} \le 64$, $1 \le \text{n\_cols} \le 256$; `BLOCK` is a power of
  two from 1 to 128; $10^{-6} \le \varepsilon \le 10^{-2}$.
- Entries of float32 inputs have magnitude at most $10^{4}$; float16 inputs at most
  2000 (whose square, $4 \times 10^{6}$, is not representable in float16). Gains are
  between 0.5 and 1.5.
- The result is compared with the formula evaluated in float64 from the values the
  kernel was given, to a relative tolerance of `1e-4` and an absolute tolerance of
  `1e-6` for float32, and `2e-3` and `1e-3` for float16.
- Limits per launch: exactly `n_rows` programs; exactly `n_rows * n_cols` elements
  stored; at most `3 * n_rows * n_cols` elements loaded (`x` in each of two passes,
  the gain once).
- Stay inside the part of Triton shown above, plus `tl.where`, `tl.rsqrt`, `tl.full`,
  comparisons, and arithmetic. No Python lists, NumPy, or torch inside the kernel.
- **What runs where.** The cases without a hardware mark run on the CPU, where
  `tl` is the kit's `tritonshim`: it executes the kernel's source in NumPy, one
  program after another, and is strict where a GPU is silent. An unmasked access
  outside a tensor raises `tl.OutOfBounds`, a masked-off value loaded without
  `other=` is NaN, two programs storing to one element raise `tl.DataRace`, and a
  non-contiguous tensor argument is refused. Two cases run only on a machine with an
  NVIDIA GPU: there `rms_norm` receives CUDA tensors, and the launch hands the same
  kernel source to the real Triton compiler. That path, and the shim's agreement
  with Triton, could not be tested when the problem was authored; the kernel in the
  reference solution follows the layer-norm tutorial of Triton's documentation
  construct by construct.

## Why this matters

When a model needs an operation that no library kernel does well (a normalization
fused with its neighbors, a quantized matmul, a sampler that never leaves the
device), the options are a chain of PyTorch operations that each make a pass over
memory, or one kernel. Triton is how that kernel gets written today: `torch.compile`
emits it, and inference stacks ship hand-written Triton for exactly the layers this
track covers. Almost every such kernel has the shape of this one (one program per
row, a loop over blocks, a masked tail), and the tail is where they break. A missing
mask does not crash on a GPU. It folds a few elements of the next row into this
row's statistics, or reads whatever follows the tensor in the allocator's pool, and
the output is wrong by an amount that depends on the batch. An accumulator left in
float16 is the other classic: the squares overflow at 256, the norm becomes
infinite, and the layer outputs zeros for exactly the activations that were large.
