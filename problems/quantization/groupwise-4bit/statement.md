Quantize a weight matrix to `bits`-bit codes, where every run of `group_size`
consecutive inputs in a row gets its own scale and bias. This is the layout of 4-bit MLX
checkpoints. The arithmetic is three lines; the work is in cutting the matrix into the
right groups, counting steps and not codes, and surviving a group whose values are all
equal.

## Specification

```python
def quantize_groups(w, bits, group_size): ...  # -> (codes, scale, bias)
```

`w` is the weight of a linear layer, shape `(out, in)`: row $i$ produces output $i$, and
column $j$ multiplies input $j$. Groups run **along the input axis**. Row $i$ is cut
into `in / group_size` consecutive groups, and group $g$ of row $i$ holds the elements

$$
w_{i,\,j} \quad \text{for} \quad g \cdot \text{group\_size} \;\le\; j \;<\; (g + 1) \cdot \text{group\_size}.
$$

Each group is quantized on its own. With $L = 2^{\text{bits}} - 1$, the largest code:

$$
\text{bias}_{i,g} = \min_{j \in g} w_{i,j},
\qquad
\text{scale}_{i,g} = \begin{cases}
\dfrac{\max_{j \in g} w_{i,j} - \min_{j \in g} w_{i,j}}{L} & \text{if the maximum exceeds the minimum}, \\[1.5ex]
1 & \text{if every element of the group is equal},
\end{cases}
$$

$$
\text{code}_{i,j} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{w_{i,j} - \text{bias}_{i,g}}{\text{scale}_{i,g}}\Big),\ 0,\ L\Big),
\qquad g = \lfloor j / \text{group\_size} \rfloor .
$$

The weights are recovered, approximately, as

$$
\hat w_{i,j} = \text{code}_{i,j} \cdot \text{scale}_{i,g} + \text{bias}_{i,g} .
$$

- `w`: a float32 matrix. `in` is a multiple of `group_size`.
- `bits`: a Python int, $2 \le \text{bits} \le 8$. The codes are unsigned:
  $0, 1, \dots, 2^{\text{bits}} - 1$.
- `group_size`: a Python int, at least 1. It may equal `in` (one group per row) or 1.
- Returns a tuple `(codes, scale, bias)`: `codes` is a **uint8** tensor with the shape
  of `w`; `scale` and `bias` are float32 tensors of shape `(out, in / group_size)`,
  one entry per group, in row order.

**Constant groups.** A group whose elements are all equal has no range. It gets
`scale = 1`, `bias` equal to its value, and codes that are all 0, so that it is
recovered exactly. No output may ever be NaN or infinite.

**Rounding.** $\operatorname{round}$ is rounding to the nearest integer. Ties go to the
even integer, which is what `round` does in PyTorch, JAX, and MLX. No test contains a
value of $(w - \text{bias}) / \text{scale}$ within $0.05$ of a tie, so a one-bit
difference in a float32 division cannot change a code. Truncating is not rounding and
does fail.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)` | float32 |
| `codes` | `(out, in)` | uint8 |
| `scale` | `(out, in / group_size)` | float32 |
| `bias` | `(out, in / group_size)` | float32 |

## Example

```python
w = [[ 0.0,  0.33, -0.5,  1.0,   2.0,  2.2,  2.9,  2.6],
     [-1.0, -1.0,  -1.0, -1.0,   0.5, -0.25, 0.1,  0.3]]

codes, scale, bias = quantize_groups(w, bits=4, group_size=4)

codes  # [[5, 8, 0, 15,   0, 3, 15, 10],
       #  [0, 0, 0,  0,  15, 0,  7, 11]]
scale  # [[0.1, 0.06],
       #  [1.0, 0.05]]
bias   # [[-0.5,  2.0 ],
       #  [-1.0, -0.25]]
```

Each row has two groups of four, and 4 bits give $L = 15$ steps.

The first group of row 0 spans $[-0.5, 1.0]$, so $\text{scale} = 1.5 / 15 = 0.1$ and
$\text{bias} = -0.5$:

| $w$ | $(w - \text{bias}) / \text{scale}$ | code | $\text{code} \cdot \text{scale} + \text{bias}$ |
|---|---|---|---|
| $0.0$ | $5.0$ | $5$ | $0.0$ |
| $0.33$ | $8.3$ | $8$ | $0.3$ |
| $-0.5$ | $0.0$ | $0$ | $-0.5$ |
| $1.0$ | $15.0$ | $15$ | $1.0$ |

The second group of row 0 spans $[2.0, 2.9]$. It never comes near zero, and it does not
have to: its bias is $2.0$ and all 16 codes are spent on a range of $0.9$
($\text{scale} = 0.06$). The value $2.2$ sits at $3.33$ steps, gets code $3$, and comes
back as $2.18$.

The first group of row 1 is constant. Its range is 0, and dividing by a scale of 0
would produce NaN; instead the scale is 1, the codes are 0, and the bias alone
reproduces $-1.0$.

The minimum and the maximum of every group come back exactly (codes $0$ and $L$).
Everything else comes back within half a step, $\text{scale} / 2$, of its group.

## Constraints

- $1 \le \text{out} \le 64$, $1 \le \text{in} \le 512$, and `in` is a multiple of
  `group_size`.
- Every entry of `w` is finite with $|w_{i,j}| \le 100$. A group that is not constant
  has a range of at least $10^{-3}$.
- `codes` are compared **exactly** and must have dtype uint8. `scale` and `bias` are
  compared to a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`.
- No Python loops over elements, groups, or rows are needed.

## Why this matters

A 4-bit model is not "every weight rounded to one of 16 values". One grid for a whole
matrix, or even for a whole row, is set by the few largest weights, and the bulk of
the values, which are much smaller, would share two or three codes. Group-wise formats
give every short run of weights its own grid: MLX's affine quantization uses groups of
64 inputs with a scale and a bias each, and GGUF's `Q4_1` uses blocks of 32 with a
scale and a minimum. The price is the per-group metadata. Two float16 numbers per 64
weights add half a bit, which is why a "4-bit" MLX model costs 4.5 bits per weight.
The grouping has to match the layout the matmul kernel expects, along the input axis,
because the kernel walks a row of codes and switches scale every `group_size` inputs.
A quantizer that groups along the wrong axis, or shares one range across a row, still
produces a file that loads and a model that talks. It is just worse, and nothing
crashes to tell you.
