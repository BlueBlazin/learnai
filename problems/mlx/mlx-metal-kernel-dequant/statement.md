Write the kernel a quantized runtime is built around: one Metal function that turns
the packed integer codes of an MLX checkpoint back into float weights, one GPU thread
per weight. Each thread has to find its code inside a 32-bit word, find the scale and
bias of its group, and do it with integer arithmetic that is right for 2, 4, and 8
bits. And it has to be as fast as the kernel MLX ships.

## Specification

```python
def metal_dequantize(words, scales, biases, group_size, bits): ...
```

The inputs are one affine-quantized matrix in the layout of `mx.quantize` and of
every MLX checkpoint (`<name>.weight`, `<name>.scales`, `<name>.biases`). The matrix
has `out` rows and `in` columns.

- `words`: uint32, shape `(out, in * bits / 32)`. A word holds `32 / bits` codes,
  the first in its lowest bits: code $j$ of a word is bits $j \cdot \text{bits}$ to
  $(j + 1) \cdot \text{bits} - 1$. A row of words therefore holds the `in` codes of
  that row in order: with $p = 32 / \text{bits}$,

  $$
  \text{code}_{r,c} = \left\lfloor \frac{\text{words}_{r,\lfloor c / p \rfloor}}{2^{\,\text{bits} \cdot (c \bmod p)}} \right\rfloor \bmod 2^{\text{bits}} .
  $$

- `scales`, `biases`: shape `(out, in / group_size)`, both float16, both bfloat16,
  or both float32. Every run of `group_size` consecutive columns of a row is a group
  with one scale and one bias.
- `group_size`: 32, 64, or 128. `bits`: 2, 4, or 8. Both are Python ints.
- Returns the dequantized matrix, shape `(out, in)`, with **the dtype of `scales`**:

  $$
  w_{r,c} = \text{code}_{r,c} \cdot \text{scales}_{r,\lfloor c / \text{group\_size} \rfloor}
  + \text{biases}_{r,\lfloor c / \text{group\_size} \rfloor} .
  $$

  The product and the sum are computed in float32 and the result is converted to the
  output dtype once. (Metal has no arithmetic for bfloat16, and half precision cannot
  hold the product of an 8-bit code and a scale exactly.)

Any of the three inputs may be a strided view, for example every second row of a
layer.

![The thread for row 0, column 37 of the worked example. It reads one word and the scale and bias of its group, and writes one weight.](assets/one-thread.svg)

**The rules of the exercise.**

1. The result is the output of one kernel built with `mx.fast.metal_kernel`. As in
   `mlx-metal-kernel-elementwise`, the judge reads the lazy graph behind the returned
   array: with evaluated inputs it must contain exactly one `CustomKernel` and,
   besides it, only operations that rearrange data (reshape, flatten, squeeze,
   expand_dims, transpose, slice, broadcast, `mx.contiguous`). No MLX arithmetic, no
   dtype cast, no evaluation inside the function.
2. `mx.dequantize` is not allowed.
3. **Speed.** On a `4096 x 4096` layer at 4 bits in groups of 64, the function takes
   at most 3 times as long as the reference kernel, which launches one thread per
   weight on a two-dimensional grid and runs as fast as `mx.dequantize` on this
   machine (0.5 ms against 0.43 ms).

Two more facts about `mx.fast.metal_kernel` beyond the previous problem, both of MLX
0.32.4:

- A `template` entry may be an **int** (or a bool): `template=[("T", dtype), ("BITS",
  4)]` makes `BITS` a compile-time constant of the kernel.
- When the body mentions `words_shape` (or `_strides`, `_ndim`) for an input named
  `words`, MLX adds it to the signature: `words_shape[1]` is the number of words per
  row.

The grid has three dimensions: `grid=(nx, ny, 1)` launches `nx * ny` threads, and
`thread_position_in_grid.x` and `.y` are a thread's coordinates.

## Shapes

| name | shape | dtype |
|---|---|---|
| `words` | `(out, in * bits / 32)` | uint32 |
| `scales` | `(out, in / group_size)` | float16, bfloat16, or float32 |
| `biases` | `(out, in / group_size)` | the dtype of `scales` |
| result | `(out, in)` | the dtype of `scales` |

## Example

Two rows of 64 weights at 4 bits in groups of 32: 8 words and 2 groups per row.

```python
words = mx.array([[0x76543210, 0xFEDCBA98] * 4, [0x00000000, 0xFFFFFFFF] * 4], dtype=mx.uint32)
scales = mx.array([[0.5, -0.25], [1.0, 0.125]])
biases = mx.array([[1.0, 2.0], [-3.0, 0.5]])

w = metal_dequantize(words, scales, biases, group_size=32, bits=4)  # shape (2, 64)
w[0, 5]  # 3.5
w[0, 37]  # 0.75
w[1, 7]  # -3.0
w[1, 63]  # 2.375
```

Row 0 holds the codes `0, 1, ..., 15` four times over; row 1 holds eight zeros, then
eight fifteens, four times over.

- `w[0, 5]`: word $\lfloor 5 / 8 \rfloor = 0$ is `0x76543210`; code $5 \bmod 8 = 5$
  is bits 20 to 23, which is 5. Group $\lfloor 5 / 32 \rfloor = 0$:
  $5 \cdot 0.5 + 1.0 = 3.5$.
- `w[0, 37]`: word 4 is `0x76543210` again, code 5 of it is 5. Group 1:
  $5 \cdot (-0.25) + 2.0 = 0.75$ (Figure 1).
- `w[1, 7]`: word 0 of row 1 is `0x00000000`, code 0. Group 0 of **row 1**:
  $0 \cdot 1.0 - 3.0 = -3.0$.
- `w[1, 63]`: word 7 is `0xFFFFFFFF`, code 7 of it is 15. Group 1 of row 1:
  $15 \cdot 0.125 + 0.5 = 2.375$.

## Constraints

- $1 \le \text{out} \le 4096$; `in` is a multiple of `group_size`,
  $32 \le \text{in} \le 4096$.
- The result is compared with $\text{code} \cdot \text{scale} + \text{bias}$ computed
  in float64 from the values the function received, with a relative and an absolute
  tolerance of `1e-5` and `1e-6` for float32, `2e-3` and `1e-3` for float16, and
  `1e-2` and `4e-3` for bfloat16 (one rounding to the output dtype is well inside
  each). Weights in the tests are of order 0.1 to 1.
- The speed limit is a ratio to the reference solution on the same machine in the
  same run; no absolute time is judged.
- Every case needs the Apple GPU.

## Why this matters

A 4-bit model is a file of packed words, and everything that touches its weights
starts by doing what this kernel does: `mx.dequantize`, the unpacking inside
`mx.quantized_matmul`, a LoRA merge that needs the float matrix for one layer, a
converter between quantization formats. When the format is one MLX does not ship (a
different packing, a codebook, per-channel scales), you write this kernel yourself,
and its two halves fail differently. The index arithmetic fails silently: reversed
nibbles, a group index that forgets the row, or a signed shift give a matrix of
plausible numbers and a model that produces fluent nonsense. The launch geometry
fails in speed: a kernel with one thread per row is correct and several times slower
on a `4096 x 4096` layer, and looping over rows in Python is slower again by another
order of magnitude.
