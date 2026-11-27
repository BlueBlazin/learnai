Implement a custom Metal kernel in MLX that unpacks and dequantizes an affine-quantized weight matrix into floating-point weights, launching one GPU thread per output weight. Each thread extracts its integer code from a packed 32-bit word, looks up its group's scale and bias, and computes the weight for 2-bit, 4-bit, or 8-bit quantization.

## Specification

```python
def metal_dequantize(words, scales, biases, group_size, bits): ...
```

The inputs represent one affine-quantized matrix with `out` rows and `in` columns in the layout used by `mx.quantize` and MLX checkpoints (`<name>.weight`, `<name>.scales`, `<name>.biases`). Both `group_size` (32, 64, or 128) and `bits` (2, 4, or 8) are Python ints.

Let $p = 32 / \text{bits}$ be the number of integer codes packed into each `uint32` word of `words`. Inside a word, code $j$ occupies bits $j \cdot \text{bits}$ through $(j + 1) \cdot \text{bits} - 1$, starting from the lowest bits. Across a row, the packed words store the row's `in` codes in column order. For row $r$ and column $c$, the unpacked unsigned code is

$$
\text{code}_{r,c} = \left\lfloor \frac{\text{words}_{r,\lfloor c / p \rfloor}}{2^{\,\text{bits} \cdot (c \bmod p)}} \right\rfloor \bmod 2^{\text{bits}} .
$$

Each row is divided into consecutive groups of `group_size` columns. Every group shares one entry in `scales` and one entry in `biases`. The dequantized weight at row $r$ and column $c$ is

$$
w_{r,c} = \text{code}_{r,c} \cdot \text{scales}_{r,\lfloor c / \text{group\_size} \rfloor}
+ \text{biases}_{r,\lfloor c / \text{group\_size} \rfloor} .
$$

Compute the product and sum in float32, then convert the result once to the dtype of `scales`. Metal does not support arithmetic directly on bfloat16, and float16 cannot represent the product of an 8-bit code and a scale without intermediate rounding loss.

Any of `words`, `scales`, or `biases` may be a strided view, such as every second row of a layer.

![The thread for row 0, column 37 of the worked example. It reads one word and the scale and bias of its group, and writes one weight.](assets/one-thread.svg)

**Exercise rules:**

1. Return the output of a single kernel built with `mx.fast.metal_kernel`. Given evaluated inputs, the lazy graph behind the returned array must contain exactly one `CustomKernel` node. Only data-rearranging operations (`reshape`, `flatten`, `squeeze`, `expand_dims`, `transpose`, `slice`, `broadcast`, `mx.contiguous`) may appear alongside it. Do not use MLX arithmetic, dtype casts, or evaluation inside the function.
2. Calling `mx.dequantize` is forbidden.
3. **Speed:** On a `4096 x 4096` layer at 4 bits with `group_size=64`, your function must run within 3x the time of the reference kernel. The reference launches one thread per weight on a 2D grid and matches `mx.dequantize` on this machine (0.50 ms vs. 0.43 ms).

**Relevant `mx.fast.metal_kernel` features (MLX 0.32.4):**

- A `template` entry can be an `int` or `bool`. Passing `template=[("T", dtype), ("BITS", 4)]` makes `BITS` a compile-time constant in the Metal source.
- Referencing `words_shape` (or `_strides`, `_ndim`) for an input named `words` automatically adds that parameter to the generated kernel signature. For example, `words_shape[1]` gives the number of words per row.
- Launching with `grid=(nx, ny, 1)` dispatches `nx * ny` threads whose coordinates are `thread_position_in_grid.x` and `thread_position_in_grid.y`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `words` | `(out, in * bits / 32)` | uint32 |
| `scales` | `(out, in / group_size)` | float16, bfloat16, or float32 |
| `biases` | `(out, in / group_size)` | same as `scales` |
| result | `(out, in)` | same as `scales` |

## Example

Consider 2 rows of 64 weights quantized to 4 bits with `group_size=32`, which gives 8 words and 2 groups per row.

```python
words = mx.array([[0x76543210, 0xFEDCBA98] * 4, [0x00000000, 0xFFFFFFFF] * 4], dtype=mx.uint32)
scales = mx.array([[0.5, -0.25], [1.0, 0.125]])
biases = mx.array([[1.0, 2.0], [-3.0, 0.5]])

w = metal_dequantize(words, scales, biases, group_size=32, bits=4)  # shape (2, 64)
w[0, 5]   # 3.5
w[0, 37]  # 0.75
w[1, 7]   # -3.0
w[1, 63]  # 2.375
```

Row 0 repeats the 4-bit codes `0, 1, ..., 15` four times. Row 1 alternates eight zeros and eight fifteens four times.

- `w[0, 5]`: word $\lfloor 5 / 8 \rfloor = 0$ is `0x76543210`. Code $5 \bmod 8 = 5$ occupies bits 20 to 23, which equals 5. In group $\lfloor 5 / 32 \rfloor = 0$ of row 0, $5 \cdot 0.5 + 1.0 = 3.5$.
- `w[0, 37]`: word $\lfloor 37 / 8 \rfloor = 4$ is `0x76543210`, and code $37 \bmod 8 = 5$ is 5. In group 1 of row 0, $5 \cdot (-0.25) + 2.0 = 0.75$ (shown in the figure above).
- `w[1, 7]`: word 0 of row 1 is `0x00000000`, so the code is 0. In group 0 of row 1, $0 \cdot 1.0 - 3.0 = -3.0$.
- `w[1, 63]`: word 7 of row 1 is `0xFFFFFFFF`, so code 7 is 15. In group 1 of row 1, $15 \cdot 0.125 + 0.5 = 2.375$.

## Constraints

- $1 \le \text{out} \le 4096$ and $32 \le \text{in} \le 4096$, where `in` is a multiple of `group_size`.
- Outputs are checked against $\text{code} \cdot \text{scale} + \text{bias}$ computed in float64 from the received inputs. Relative and absolute tolerances are `1e-5` and `1e-6` for float32, `2e-3` and `1e-3` for float16, and `1e-2` and `4e-3` for bfloat16. A single rounding to the output dtype stays well within each bound. Test weights have magnitude around 0.1 to 1.
- Speed is judged strictly as a ratio against the reference solution on the same machine in the same run, with no fixed wall-clock threshold.
- All test cases require an Apple GPU.

## Why this matters

Quantized checkpoints store weights as packed integer words. Every operation that touches those weights—`mx.dequantize`, the unpacking stage inside `mx.quantized_matmul`, merging a LoRA adapter into a base layer, or converting between quantization formats—relies on this unpacking logic. When you need a custom format that MLX does not ship, such as a different packing layout, a codebook, or per-channel scales, you write this kernel yourself.

Bit-indexing bugs fail silently. Reversed bit order, a group index that drops the row offset, or a signed right shift still produce finite, plausible-looking floats that cause a model to generate fluent nonsense. Launch-geometry bugs hurt performance instead. Launching one thread per row rather than one thread per weight produces correct values but runs several times slower on a `4096 x 4096` layer, and looping over rows in Python is another order of magnitude slower.
