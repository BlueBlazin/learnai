Write the kernel a 4-bit model actually runs on: one Triton program per output element
that reads packed codes, unpacks them with a shift and a mask, applies the scale and
bias of each column's group, and accumulates the dot product with the activations,
all in registers. The dequantized weight matrix is never written anywhere. Three
index calculations per column have to agree (word, nibble, group), and nothing but
the numbers tells you when one is off.

## Specification

```python
from learnai.kit import tritonshim as tl


@tl.jit
def qlinear_kernel(
    x_ptr, words_ptr, scales_ptr, biases_ptr, y_ptr, n_in, n_out, group_size, BLOCK: tl.constexpr
): ...


def qlinear(x, words, scales, biases, group_size, block=64): ...
```

**The quantized matrix** has `n_out` rows and `n_in` columns, in the layout of
`bit-packing` and of MLX checkpoints at 4 bits:

- `words`: uint32, shape `(n_out, n_in / 8)`. A word holds eight 4-bit codes, the
  first in its lowest bits:

  $$
  \text{code}_{o,c} = \left\lfloor \frac{\text{words}_{o,\lfloor c / 8 \rfloor}}{16^{\,c \bmod 8}} \right\rfloor \bmod 16 .
  $$

- `scales`, `biases`: shape `(n_out, n_in / group_size)`. Every run of `group_size`
  consecutive columns of a row is a group with one scale and one bias, and the
  weight the codes stand for is

  $$
  \hat W_{o,c} = \text{code}_{o,c} \cdot \text{scales}_{o,g} + \text{biases}_{o,g},
  \qquad g = \lfloor c / \text{group\_size} \rfloor .
  $$

**`qlinear_kernel`** is launched on a two-dimensional grid of `(n_out, batch)`.
Program `(o, b)`, with `o = tl.program_id(0)` and `b = tl.program_id(1)`, computes one
element of $y = x \hat W^{\top}$ and stores it:

$$
y_{b,o} = \sum_{c=0}^{\text{n\_in}-1} x_{b,c} \, \hat W_{o,c} .
$$

- All five tensors arrive as pointers to row-major memory: `x` is `(batch, n_in)`,
  `y` is `(batch, n_out)`, so $y_{b,o}$ is at offset `b * n_out + o`.
- `x`, `scales`, and `biases` are float32 or float16 (`scales` and `biases` always
  have the same type). Every product and the sum are computed in **float32**; the
  store converts the sum to the element type of `y`.
- `n_in`, `n_out`, and `group_size` arrive as int32 scalars; `BLOCK`, the number of
  columns processed at a time, is a compile-time power of two. `n_in` need not be a
  multiple of `BLOCK`, and `group_size` and `BLOCK` are unrelated: a block may cover
  parts of several groups, and a group may span several blocks.
- **The kernel stores exactly one element, $y_{b,o}$, and nothing else.** There is
  no buffer for dequantized weights; codes and weights exist only as blocks inside
  the program. It reads no element outside the tensors.

The language is the one introduced in `cuda-triton-rmsnorm`. New here:
`tl.program_id(1)` for the second grid axis; the integer operators `//`, `%`, `>>`,
`<<`, and `&` on blocks; the unsigned type `tl.uint32`; and loads through computed
offsets such as `ptr + cols // 8`, which read one element per entry of the block
even when entries repeat.

![Output row 0 and batch row 0 of the worked example, column by column. The weights in the dashed row are never stored.](assets/one-program.svg)

**`qlinear(x, words, scales, biases, group_size, block)`** applies the kernel.

- `x`: float32 or float16, shape `(..., n_in)` with one to three dimensions. Any of
  the four tensors may be a strided view.
- It launches `qlinear_kernel` **once**, on a grid of `(n_out, batch)` where `batch`
  is the number of vectors along the last axis of `x`, with `BLOCK=block`.
- It returns a new tensor of shape `(..., n_out)` with the dtype and device of `x`.
  It builds no dequantized matrix of its own.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., n_in)`; `(batch, n_in)` behind `x_ptr` | float32 or float16 |
| `words` | `(n_out, n_in / 8)` | uint32 |
| `scales`, `biases` | `(n_out, n_in / group_size)` | float32 or float16 |
| memory behind `y_ptr` | `(batch, n_out)` | float32 or float16 |
| result of `qlinear` | `(..., n_out)` | the dtype of `x` |

## Example

The layer of `quantized-matmul`'s example, packed: two rows of eight columns in
groups of four, so one word and two groups per row.

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

Read `0xAF30F085` from the right: `5, 8, 0, F, 0, 3, F, A`. Program `(0, 0)` loads
one block of 16 columns of which 8 exist. For column 3 it finds word $3 // 8 = 0$,
shifts it right by $4 \cdot (3 \bmod 8) = 12$ bits and masks with `0xF`: code 15.
Its group is $3 // 4 = 0$, so the weight is $15 \cdot 0.1 - 0.5 = 1.0$, and the
product with $x_{0,3} = 0.5$ is $0.5$. Over the eight columns the weights are
$[0.0, 0.3, -0.5, 1.0, 2.0, 2.18, 2.9, 2.6]$ and the sum of products is
$0.6 + 0.5 + 0.5 + 2.18 + 2.9 - 2.6 = 4.08$ (Figure 1).

The second batch row is zero except in column 4, so it reads off single weights:
$\hat W_{0,4} = 0 \cdot 0.06 + 2.0$ and $\hat W_{1,4} = 15 \cdot 0.05 - 0.25 = 0.5$.
Column 4 of row 1 is the code 15, hexadecimal digit `F` of `0xB70F0000`: a kernel
that sign-extends it reads $-1$ and returns $-0.3$.

## Constraints

- $1 \le \text{n\_out} \le 16$, $1 \le \text{batch} \le 8$, $8 \le \text{n\_in} \le 256$;
  `n_in` is a multiple of 8 and of `group_size`; $1 \le \text{group\_size} \le
  \text{n\_in}$; `BLOCK` is a power of two from 1 to 128.
- Weights are of order 0.1. Activations are of order 1 in most cases; float16
  activations reach a few thousand.
- The result is compared with $x \hat W^{\top}$ evaluated in float64 from the values
  the kernel was given, to a relative tolerance of `1e-4` and an absolute tolerance
  of `1e-4` for a float32 output (`2e-3` and `2e-3` for float16). Where the products
  are large and cancel, the absolute tolerance is $2 \times 10^{-6}$ times the
  largest $\sum_c |x_{b,c} \hat W_{o,c}|$ of the case, if that is larger.
- Limits per launch: exactly `n_out * batch` programs; exactly one element stored per
  program; at most `4 * n_in` elements loaded per program (per column: an
  activation, a word, a scale, a bias). The inputs are read-only.
- **What runs where.** The cases without a hardware mark run on the CPU under the
  kit's `tritonshim`, which is strict as in `cuda-triton-rmsnorm`: out-of-bounds
  accesses, undefined masked values, races between programs, and non-contiguous
  arguments are errors. Two cases run only on a machine with an NVIDIA GPU, where
  `qlinear` receives CUDA tensors and the same kernel source is compiled by the real
  Triton. That path could not be tested when the problem was authored. In
  particular, the documentation of Triton does not spell out what `>>` does. The
  shim shifts zeros into an unsigned value, and refuses to shift a negative signed
  value, because it cannot know whether the sign bit would be copied.

## Why this matters

Quantization pays only if the float weights never come back. A 4-bit matrix is an
eighth of its float32 size; dequantize it before the matmul and the layer needs
nine eighths, and the memory bandwidth that bounds tokens per second is spent on
weights again. So engines fuse: MLX's `quantized_matmul` and llama.cpp's quantized
dot products unpack inside the inner loop, and so do the hand-written CUDA kernels
that ship with other 4-bit formats. When a new format appears (another packing, per-channel scales, a codebook),
this is the kernel that has to be written for it, and its bugs are quiet ones.
Reversed nibbles, the scale of the neighboring group, or a sign-extended code each
produce a matrix of plausible numbers and a model that is fluent and wrong; a
float16 accumulator produces one that is right on short inputs and drifts on long
ones.
