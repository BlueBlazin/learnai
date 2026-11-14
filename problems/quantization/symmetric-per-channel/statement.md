Quantize a weight matrix symmetrically, with one scale per output row: each row is
mapped to signed integer codes on its own grid, centered on zero, with no zero-point.
The formula is one line. The work is in the axes (the scale belongs to a row, and must
broadcast along it) and in the range, which is deliberately one code short of what the
integer type can hold.

## Specification

```python
def quantize_weight(w, bits): ...  # -> (q, scale)
def dequantize(q, scale): ...  # -> float32 weights
```

`w` is the weight of a linear layer, of shape `(out, in)`: row $i$ holds the weights
that produce output $i$. A symmetric grid represents the real number $r$ by the code
$q$ with $r = \text{scale} \cdot q$. With `bits` bits the codes are

$$
q \in \{-q_{\max}, \dots, q_{\max}\}, \qquad q_{\max} = 2^{\,\text{bits} - 1} - 1 ,
$$

so 8 bits use $[-127, 127]$ and 4 bits use $[-7, 7]$. The most negative
two's-complement value, $-2^{\,\text{bits}-1}$, is never used: the grid has the same
number of codes on both sides of zero.

**`quantize_weight(w, bits)`** computes, for every row $i$ separately,

$$
m_i = \max_j |w_{ij}|, \qquad
\text{scale}_i = \begin{cases}
m_i / q_{\max} & m_i > 0, \\
1 & m_i = 0 \ \text{(the row is all zeros)},
\end{cases}
$$

$$
q_{ij} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{w_{ij}}{\text{scale}_i}\Big),\ -q_{\max},\ q_{\max}\Big).
$$

It returns the pair `(q, scale)`: `q` as an **int8** tensor of shape `(out, in)`,
whatever `bits` is, and `scale` as a float32 tensor of shape `(out,)`. The element of
largest magnitude in a row maps to $+q_{\max}$ or $-q_{\max}$, according to its sign.
An all-zero row gets scale 1 and codes 0; nothing may become NaN or infinite.

**`dequantize(q, scale)`** returns the float32 tensor of shape `(out, in)` with

$$
\hat{w}_{ij} = \text{scale}_i \cdot q_{ij} .
$$

`q` is an int8 tensor of shape `(out, in)` and `scale` a float32 tensor of shape
`(out,)`, as `quantize_weight` returns them. `scale` has one entry per **row** of `q`.

**Rounding.** $\operatorname{round}$ is rounding to the nearest integer, with ties to
the even integer, which is what `round` does in PyTorch, JAX, and MLX. The tests
contain no value of $w_{ij} / \text{scale}_i$ within $0.05$ of a tie, so a one-bit
difference in a float32 division cannot change a code. Truncation is not rounding.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)` | float32 |
| `bits` | Python `int`, $2 \le \text{bits} \le 8$ | |
| `q` | `(out, in)` | int8 |
| `scale` | `(out,)` | float32 |
| result of `dequantize` | `(out, in)` | float32 |

## Example

```python
w = [[0.4, -1.0, 0.25, 0.8],
     [0.02, 0.01, -0.03, 0.0],
     [0.0, 0.0, 0.0, 0.0]]

q, scale = quantize_weight(w, 8)
# scale = [0.00787402, 0.00023622, 1.0]
# q = [[51, -127, 32, 102],
#      [85, 42, -127, 0],
#      [0, 0, 0, 0]]

dequantize(q, scale)
# [[0.401575, -1.0, 0.251969, 0.803150],
#  [0.020079, 0.009921, -0.03, 0.0],
#  [0.0, 0.0, 0.0, 0.0]]
```

Row 0 has $m_0 = 1.0$, so $\text{scale}_0 = 1 / 127$ and its codes are
$\operatorname{round}(127 \cdot [0.4, -1.0, 0.25, 0.8]) =
\operatorname{round}([50.8, -127, 31.75, 101.6])$. Row 1 is about fifty times smaller,
$m_1 = 0.03$, and gets its own scale $0.03 / 127$: its codes
$\operatorname{round}([84.67, 42.33, -127, 0])$ use the whole range as well. With a
single scale for the tensor, $1 / 127$, row 1 would have been quantized to
`[3, 1, -4, 0]`. Row 2 is all zeros: scale 1, codes 0.

With `bits = 4`, $q_{\max} = 7$: the scales are `[0.142857, 0.00428571, 1.0]` and the
codes are `[[3, -7, 2, 6], [5, 2, -7, 0], [0, 0, 0, 0]]`.

## Constraints

- $1 \le \text{out}, \text{in}$ and `w` has at most 100,000 elements. Every weight is
  finite with $|w_{ij}| \le 10^4$, and a row that is not all zeros has $m_i \ge 10^{-6}$.
- The codes are compared **exactly** and must have dtype int8. `scale` and the
  dequantized weights are compared to a relative tolerance of `1e-4` and an absolute
  tolerance of `1e-6`.
- After the round trip, every weight is within half a step of its row,
  $|\hat{w}_{ij} - w_{ij}| \le \text{scale}_i / 2$.
- No Python loop over rows or elements is needed.

## Why this matters

This is how the weights of linear and convolution layers are quantized in essentially
every integer inference stack. The grid is symmetric because a zero-point on the
weights would add a cross term to every integer matrix multiplication; with zero-point
0, the product of codes is the product of values up to one scale. It is per-channel
because the rows of a real weight matrix differ in magnitude by an order of magnitude
or more, and a single scale set by the largest row leaves the small rows with a
handful of distinct codes. And the per-row scale costs nothing at inference: it is one
multiplication per output, after the integer accumulation. The usual way to get this
wrong is an axis: a scale computed per column, or broadcast against the wrong
dimension, produces a model that runs, has the right shapes whenever the layer is
square, and has lost most of its accuracy.
