Multiply activations by a group-quantized weight matrix **directly from its codes**:
compute `x @ W_hat.T` where `W_hat = codes * scale + bias` group by group, without ever
building `W_hat`. The trick is one line of algebra; the work is keeping three things
aligned along the input axis: the inputs, the codes, and the per-group scales and
biases.

## Specification

```python
def qmatmul(x, codes, scale, bias, group_size): ...
```

The quantized matrix has `out` rows and `in` columns. Row $i$ is cut into
$G = \text{in} / \text{group\_size}$ consecutive groups along the input axis, and
group $g$ covers the inputs $j$ with
$g \cdot \text{group\_size} \le j < (g + 1) \cdot \text{group\_size}$. Each group has
one scale and one bias, and the weight that the codes stand for is

$$
\hat W_{i,j} = \text{codes}_{i,j} \cdot \text{scale}_{i,g} + \text{bias}_{i,g},
\qquad g = \lfloor j / \text{group\_size} \rfloor .
$$

`qmatmul` returns $y = x \hat W^{\top}$, that is, for every leading index of `x`,

$$
y_i \;=\; \sum_{j} x_j \, \hat W_{i,j}
\;=\; \sum_{g=0}^{G-1} \Big( \text{scale}_{i,g} \sum_{j \in g} x_j \, \text{codes}_{i,j}
\;+\; \text{bias}_{i,g} \sum_{j \in g} x_j \Big).
$$

- `x`: float32 activations of shape `(..., in)` with one to three dimensions. The last
  axis is the input axis; all other axes are batch axes.
- `codes`: a uint8 tensor of shape `(out, in)`. Any value from 0 to 255 may appear.
- `scale`, `bias`: float32 tensors of shape `(out, in / group_size)`. Entry `[i, g]`
  belongs to group $g$ of row $i$. Scales may be negative.
- `group_size`: a Python int that divides `in`. It may equal `in` (one group per row)
  or 1.
- Returns a float32 tensor of shape `(..., out)`.

**The rule of the exercise.** Compute the result from the right-hand side of the
formula: dot products of inputs with raw codes, scaled per group, plus each bias times
the sum of its group's inputs. Do not materialize the dequantized matrix
`codes * scale + bias` and multiply by it. Converting the codes to float32 for the
dot products is allowed.

> [!NOTE]
> The judge compares outputs, and the two ways of computing them agree, so it cannot
> see which one you used. The rule is the point of the problem all the same: the GPU
> kernel problems later in the catalog compute the same sum from packed codes, and
> there the float matrix does not fit in the memory you are given.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., in)`, one to three dimensions | float32 |
| `codes` | `(out, in)` | uint8 |
| `scale` | `(out, in / group_size)` | float32 |
| `bias` | `(out, in / group_size)` | float32 |
| result | `(..., out)` | float32 |

## Example

```python
codes = [[5, 8, 0, 15,   0, 3, 15, 10],
         [0, 0, 0,  0,  15, 0,  7, 11]]
scale = [[0.1, 0.06],
         [1.0, 0.05]]
bias  = [[-0.5,  2.0 ],
         [-1.0, -0.25]]
x = [1.0, 2.0, -1.0, 0.5,   0.0, 1.0, 1.0, -1.0]

qmatmul(x, codes, scale, bias, group_size=4)  # [4.08, -2.95]
```

There are two groups of four inputs. Everything that depends on `x` is computed once
per group:

| | group 0 | group 1 |
|---|---|---|
| inputs $x_g$ | $[1, 2, -1, 0.5]$ | $[0, 1, 1, -1]$ |
| $\sum x_g$ | $2.5$ | $1$ |
| $x_g \cdot \text{codes}_{0,g}$ | $5 + 16 + 0 + 7.5 = 28.5$ | $0 + 3 + 15 - 10 = 8$ |
| $x_g \cdot \text{codes}_{1,g}$ | $0$ | $0 + 0 + 7 - 11 = -4$ |

Then each output combines its own scales and biases with those numbers:

$$
y_0 = 0.1 \cdot 28.5 + (-0.5) \cdot 2.5 \;+\; 0.06 \cdot 8 + 2.0 \cdot 1 = 4.08,
$$

$$
y_1 = 1.0 \cdot 0 + (-1.0) \cdot 2.5 \;+\; 0.05 \cdot (-4) + (-0.25) \cdot 1 = -2.95 .
$$

The same numbers come out of multiplying `x` by the dequantized matrix
`[[0.0, 0.3, -0.5, 1.0, 2.0, 2.18, 2.9, 2.6], [-1.0, -1.0, -1.0, -1.0, 0.5, -0.25, 0.1, 0.3]]`,
which is the matrix this function must never build.

## Constraints

- $1 \le \text{out} \le 64$, $1 \le \text{in} \le 256$, and `in` is a multiple of
  `group_size`. `x` has at most 64 rows in total over its batch axes.
- $|x_j| \le 10$, $|\text{scale}| \le 1$, and $|\text{bias}| \le 10$.
- The result is compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `5e-4`. The absolute part allows for float32 rounding in sums
  of up to 256 products that partly cancel.
- No Python loops are needed.

## Why this matters

A quantized model that is dequantized before every matmul has saved disk space and
nothing else: the float weights are back in memory, and memory bandwidth, which bounds
tokens per second on every machine that runs these models, is spent on them again. So
inference engines multiply from the codes. MLX's `mx.quantized_matmul`, the operation
behind every `QuantizedLinear` layer, reads the packed 4-bit codes inside the kernel
and applies each group's scale and bias as it accumulates; llama.cpp's quantized dot
products do the same per block. The identity you implement here is what makes that
cheap: the codes meet the activations as small integers, the scale is applied once per
group and not once per weight, and the bias costs one sum of inputs per group that is
shared by every output row. When this is wrong, it is wrong by a bias or a scale in
some groups only, which gives a model that is fluent and subtly off.
