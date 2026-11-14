Build the layer every quantized MLX model is made of: a `QLinear` that takes a float
`nn.Linear`, keeps only its packed quantization codes with their per-group scales and
biases, and computes its output from those codes with `mx.quantized_matmul`. The
arithmetic is MLX's; the work is storing exactly what a quantized checkpoint stores,
never rebuilding the float matrix, and handling the modes where `mx.quantize` returns
two arrays and not three.

## Specification

```python
class QLinear(nn.Module):
    def __init__(self, linear, group_size=64, bits=4, mode="affine"): ...
    def __call__(self, x): ...
```

**Construction.** `linear` is an `nn.Linear` whose `weight` is a float32 array of
shape `(out, in)`; it has a `bias` of shape `(out,)` or none (`"bias" in linear`).
The constructor quantizes the weight with

```python
mx.quantize(linear.weight, group_size, bits, mode=mode)
```

which cuts every row into `in / group_size` groups of consecutive inputs and returns

- for `mode="affine"`, three arrays `(codes, scales, biases)`: the weight that the
  codes stand for is $\hat W_{i,j} = \text{code}_{i,j} \cdot \text{scale}_{i,g} +
  \text{bias}_{i,g}$ with $g = \lfloor j / \text{group\_size} \rfloor$;
- for `mode="mxfp4"` and `mode="mxfp8"`, two arrays `(codes, scales)`: each code is a
  small floating-point number, each scale a power of two stored as one byte, and there
  is no bias.

In both cases `codes` is a uint32 array of shape `(out, in * bits / 32)`: the codes of
a row packed `32 / bits` to a word, lowest bits first (Figure 1).

![One row of 64 weights at 4 bits in one group. The 256 bytes of float32 become 8 words of codes plus one scale and one bias; inside a word, code 0 occupies the lowest four bits.](assets/packed-row.svg)

After construction the layer's parameters (`layer.parameters()`) are exactly these,
under exactly these names, which are the names of the tensors in a quantized MLX
checkpoint:

| name | present | value |
|---|---|---|
| `weight` | always | the packed codes |
| `scales` | always | the scales |
| `biases` | `mode="affine"` only | the per-group biases of the quantization |
| `bias` | only if `linear` has a bias | the bias of the float layer, unchanged |

The float weight is not a parameter and is not kept anywhere else: once the caller
drops `linear`, nothing may hold its `(out, in)` float matrix. `linear` itself is not
modified. The constructor may leave its arrays lazy; the judge evaluates
`layer.parameters()`.

**Forward pass.** `layer(x)` returns

$$
y = x \hat W^{\top} + \text{bias}
$$

(without the last term when there is no bias), computed **from the packed codes** by
one call of `mx.quantized_matmul`. `x` is a float32 array of shape `(..., in)`; the
result is float32 of shape `(..., out)`.

The judge reads the lazy graph of `layer(x)` before evaluating it: it must contain
exactly one `QuantizedMatmul` operation and no `Matmul`, `Quantize`, or `Dequantize`.
A layer that dequantizes, at construction or on every call, gives the same numbers
and is rejected.

**The rule of the exercise.** Build the layer from `mx.quantize` and
`mx.quantized_matmul`. Do not use `nn.QuantizedLinear`, `nn.QQLinear`,
`nn.QuantizedEmbedding`, the `to_quantized` methods, or `nn.quantize`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `linear.weight` | `(out, in)` | float32 |
| `linear.bias` (optional) | `(out,)` | float32 |
| `weight` | `(out, in * bits / 32)` | uint32 |
| `scales` | `(out, in / group_size)` | float32 (affine), uint8 (mxfp4, mxfp8) |
| `biases` (affine) | `(out, in / group_size)` | float32 |
| `bias` (optional) | `(out,)` | float32 |
| `x` | `(..., in)` | float32 |
| result | `(..., out)` | float32 |

## Example

A layer with 64 inputs and 2 outputs, so each row is one group of 64. Row 0 repeats
`1.0, 0.9, ..., -0.5` (16 values, a step of 0.1) four times; row 1 repeats
`2.9, 2.84, ..., 2.0` (a step of 0.06) four times. The bias is `[0.5, -0.2]`.

```python
layer = QLinear(linear, group_size=64, bits=4)

layer.weight  # uint32, shape (2, 8); each row is [0x76543210, 0xFEDCBA98] four times
layer.scales  # [[-0.1], [-0.0604167]]
layer.biases  # [[1.0], [2.9]]
layer.bias  # [0.5, -0.2]

x = mx.zeros(64)
x[0] = x[1] = x[2] = 1.0
x[17] = 2.0
layer(x)  # [5.0, 13.9979]        (the float layer gives [5.0, 14.0])
```

Both rows get the codes `0, 1, ..., 15` four times over: the first word holds codes 0
to 7, code 0 in its lowest four bits, which reads `0x76543210` in hexadecimal.

Row 0 is recovered exactly, $\hat w = 1.0 - 0.1 \cdot \text{code}$, so
$y_0 = 1.0 + 0.9 + 0.8 + 2 \cdot 0.9 + 0.5 = 5.0$.

Row 1 is not. MLX starts the grid at the end of the group's range that is farther from
zero, here $2.9$, and then adjusts the step so that $0$ would be an exact multiple of
it: $2.9 / 0.06 = 48.33$ is rounded to 48 steps and the scale becomes
$-2.9 / 48 = -0.0604167$. The weights come back as $2.9, 2.83958, 2.77917, \dots$,
and

$$
y_1 = 2.9 + 2.83958 + 2.77917 + 2 \cdot 2.83958 - 0.2 = 13.9979 .
$$

The float layer stores $2 \cdot 64 \cdot 4 + 8 = 520$ bytes. `QLinear` stores
$64 + 8 + 8 + 8 = 88$ bytes: 8 words per row, one scale and one bias per row, and the
float bias.

## Constraints

- `in` is a multiple of `group_size`. The tests use the combinations `mx.quantize`
  accepts: group sizes 32, 64, and 128 with 2, 4, or 8 bits for `"affine"`; group size
  32 with 4 bits for `"mxfp4"` and with 8 bits for `"mxfp8"`.
- $1 \le \text{out} \le 2048$, $32 \le \text{in} \le 2048$; `x` has one to three
  dimensions.
- `weight` is compared exactly, and `scales`, `biases`, and `bias` to a relative
  tolerance of `1e-4`, with what `mx.quantize` returns. The output is compared with
  $x \hat W^{\top} + \text{bias}$ computed in float64 from the unpacked codes, to a
  relative tolerance of `1e-4` and an absolute tolerance of `1e-4`.
- **Size.** At 4 bits in groups of 64, the parameters take at most 0.2 of the bytes of
  the float32 layer's parameters (5 bits per weight against 32); at 8 bits, at most
  0.3. In the example, at most 88 bytes.
- **Memory**, on the Apple GPU, for a 2048 × 2048 layer whose float weight is 16 MiB:
  after the float layer is deleted, the layer holds at most 5 MiB (the reference:
  2.5 MiB), and a forward pass on 8 rows allocates at most 4 MiB at its peak (the
  reference: under 0.1 MiB).

## Why this matters

A 4-bit model runs in a quarter of the memory only if the float weights never come
back. On Apple silicon the weights live in the same unified memory as everything else,
and decoding speed is bounded by how many bytes of weights each token has to pull
through the memory bus, so a layer that dequantizes on every call pays for the float
matrix in both currencies: 16 MiB allocated, written, and read again per layer per
token, where the packed codes are 2 MiB read once. `mx.quantized_matmul` is the
operation that avoids it, and `mlx-lm` builds every quantized model out of layers
shaped like this one. The parameter names matter as much as the arithmetic: a
checkpoint written by `mlx_lm.convert` stores `weight`, `scales`, and `biases` for
each layer, and a module tree with other names, or with a float `weight`, cannot load
it.
