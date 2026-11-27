Implement a fused 4-bit dequantize-and-multiply kernel in Triton (`qlinear_kernel`) along with a PyTorch wrapper (`qlinear`). Each program computes one output element by reading packed 4-bit codes, unpacking and scaling them in registers, and accumulating the dot product with the input activations without ever writing the dequantized weight matrix to memory.

## Specification

```python
from learnai.kit import tritonshim as tl


@tl.jit
def qlinear_kernel(
    x_ptr, words_ptr, scales_ptr, biases_ptr, y_ptr, n_in, n_out, group_size, BLOCK: tl.constexpr
): ...


def qlinear(x, words, scales, biases, group_size, block=64): ...
```

### Quantized weight layout

The quantized weight matrix has `n_out` rows and `n_in` columns, using the 4-bit packed layout of `bit-packing` and MLX checkpoints:

- `words` (`uint32`, shape `(n_out, n_in / 8)`): each 32-bit word stores eight 4-bit unsigned codes, with the first code in the lowest 4 bits. For output row $o$ and input column $c$:

  $$
  \text{code}_{o,c} = \left\lfloor \frac{\text{words}_{o,\lfloor c / 8 \rfloor}}{16^{\,c \bmod 8}} \right\rfloor \bmod 16
  $$

- `scales` and `biases` (shape `(n_out, n_in / group_size)`): each consecutive group of `group_size` columns in row $o$ shares one scale and one bias. The dequantized weight is:

  $$
  \hat W_{o,c} = \text{code}_{o,c} \cdot \text{scales}_{o,g} + \text{biases}_{o,g},
  \qquad g = \lfloor c / \text{group\_size} \rfloor
  $$

### `qlinear_kernel`

Launch `qlinear_kernel` on a 2D grid of `(n_out, batch)` programs, where `batch` is the number of input vectors. Program `(o, b)`, with `o = tl.program_id(0)` and `b = tl.program_id(1)`, computes one entry of $y = x \hat W^{\top}$:

$$
y_{b,o} = \sum_{c=0}^{\text{n\_in}-1} x_{b,c} \, \hat W_{o,c}
$$

- **Memory layout.** All five tensor arguments arrive as pointers to contiguous row-major buffers. Input `x` has shape `(batch, n_in)` and output `y` has shape `(batch, n_out)`, so $y_{b,o}$ sits at offset `b * n_out + o`.
- **Precision.** `x`, `scales`, and `biases` are `float32` or `float16` (`scales` and `biases` always share the same dtype). Compute every dequantized weight, product, and partial sum in `float32`, then let `tl.store` convert the final sum to the dtype of `y`.
- **Blocking.** `n_in`, `n_out`, and `group_size` arrive as `int32` scalars, and `BLOCK` is a compile-time power of two specifying how many columns to process per iteration. `n_in` does not have to be a multiple of `BLOCK`. Moreover, `group_size` and `BLOCK` are independent. A single block may cross multiple groups, and a single group may span multiple blocks.
- **Fused execution.** Each program stores exactly one element ($y_{b,o}$) and nothing else. Never materialize a buffer for dequantized weights, and never read outside the input tensors.
- **New Triton primitives.** Beyond the subset introduced in `cuda-triton-rmsnorm`, this kernel uses `tl.program_id(1)` for the second grid axis, blockwise integer operators (`//`, `%`, `>>`, `<<`, `&`), the unsigned type `tl.uint32`, and gathered loads through computed offsets such as `ptr + cols // 8` (which read one element per lane in the block even when offsets repeat).

![Output row 0 and batch row 0 of the worked example, column by column. The weights in the dashed row are never stored.](assets/one-program.svg)

### `qlinear`

`qlinear(x, words, scales, biases, group_size, block=64)` applies `qlinear_kernel` to an input tensor `x` of shape `(..., n_in)` with one to three dimensions.

- Any of the four input tensors (`x`, `words`, `scales`, `biases`) may be a non-contiguous strided view.
- Launch `qlinear_kernel` **once** on a grid of `(n_out, batch)` (where `batch` is the total number of vectors along the last axis of `x`) with `BLOCK=block`.
- Return a new tensor of shape `(..., n_out)` with the dtype and device of `x`. Do not allocate a dequantized weight matrix in the wrapper.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., n_in)`; `(batch, n_in)` behind `x_ptr` | `float32` or `float16` |
| `words` | `(n_out, n_in / 8)` | `uint32` |
| `scales`, `biases` | `(n_out, n_in / group_size)` | `float32` or `float16` |
| memory behind `y_ptr` | `(batch, n_out)` | `float32` or `float16` |
| result of `qlinear` | `(..., n_out)` | same as `x` |

## Example

This example uses the packed version of the layer from `quantized-matmul`. It has two output rows of eight columns in groups of four, giving one word and two groups per row.

```python
words = [
    [0xAF30F085],
    [0xB70F0000],
]  # codes [5, 8, 0, 15, 0, 3, 15, 10] and [0, 0, 0, 0, 15, 0, 7, 11]
scales = [[0.1, 0.06], [1.0, 0.05]]
biases = [[-0.5, 2.0], [-1.0, -0.25]]
x = [[1.0, 2.0, -1.0, 0.5, 0.0, 1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]

qlinear_kernel[(2, 2)](x, words, scales, biases, y, 8, 2, 4, BLOCK=16)
# y = [[4.08, -2.95],
#      [2.0,   0.5 ]]
```

Reading `0xAF30F085` from least significant nibble to most significant gives `5, 8, 0, F, 0, 3, F, A`. Program `(0, 0)` processes one block of 16 columns, of which only the first 8 exist. For column 3, it reads word `3 // 8 = 0`, shifts right by `4 * (3 % 8) = 12` bits, and masks with `0xF` to get code 15. Its group is `3 // 4 = 0`, so the weight is $15 \cdot 0.1 - 0.5 = 1.0$, and its product with $x_{0,3} = 0.5$ is $0.5$. Across all eight columns, the dequantized weights are $[0.0, 0.3, -0.5, 1.0, 2.0, 2.18, 2.9, 2.6]$ and the dot product is $0.6 + 0.5 + 0.5 + 2.18 + 2.9 - 2.6 = 4.08$ (Figure 1).

The second batch row is zero everywhere except column 4, so it isolates single weights: $\hat W_{0,4} = 0 \cdot 0.06 + 2.0 = 2.0$ and $\hat W_{1,4} = 15 \cdot 0.05 - 0.25 = 0.5$. Column 4 of row 1 is code 15 (hex digit `F` in `0xB70F0000`). A kernel that sign-extends this nibble reads $-1$ instead of 15 and outputs $-0.3$.

## Constraints

- $1 \le \text{n\_out} \le 16$, $1 \le \text{batch} \le 8$, $8 \le \text{n\_in} \le 256$, and $1 \le \text{group\_size} \le \text{n\_in}$. `n_in` is a multiple of 8 and of `group_size`. `BLOCK` is a power of two from 1 to 128.
- Weights have magnitude of order 0.1. Activations are of order 1 in most cases, though `float16` activations can reach a few thousand.
- Outputs are compared against $x \hat W^{\top}$ evaluated in `float64` from the given input values. Tolerances are `rtol=1e-4`, `atol=1e-4` for `float32` outputs, and `rtol=2e-3`, `atol=2e-3` for `float16` outputs. When large products cancel, `atol` is raised to $2 \times 10^{-6}$ times the largest $\sum_c |x_{b,c} \hat W_{o,c}|$ in the test case if that is larger.
- Resource limits per launch: exactly `n_out * batch` programs, exactly 1 element stored per program, and at most `4 * n_in` elements loaded per program (one activation, word, scale, and bias per column). All input buffers are read-only.
- **CPU and GPU execution.** Standard test cases run on the CPU under `tritonshim`, enforcing the same strict checks as `cuda-triton-rmsnorm` (out-of-bounds accesses, undefined masked values, write races, and non-contiguous arguments all raise errors). Two hardware-marked cases run only on machines with an NVIDIA GPU, compiling the same kernel source with the real Triton compiler. (That GPU path could not be tested during authoring. Because Triton's documentation does not specify signed right-shift behavior, `tritonshim` shifts zeros into unsigned values and refuses to shift negative signed values.)

## Why this matters

Weight quantization saves memory bandwidth only if the full-precision weights are never written back to memory. A 4-bit weight matrix takes one eighth the space of `float32`. If you dequantize it into a temporary buffer before calling `matmul`, the layer transfers nine eighths of the `float32` traffic instead, losing the token-generation speedup.

Inference engines such as MLX (`quantized_matmul`) and llama.cpp avoid this by unpacking weights directly inside the dot-product loop, and custom CUDA or Triton kernels do the same whenever a new quantization format appears. Bugs in these kernels are silent. Reversed nibbles, an off-by-one group index, or a sign-extended code all produce plausible floating-point matrices that make a language model fluent yet subtly wrong. Likewise, accumulating in `float16` looks accurate on short hidden dimensions but drifts on long ones.
