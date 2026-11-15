Quantize a weight matrix to the three values $-1$, $0$, and $+1$ with one scale for the
whole matrix, as BitNet b1.58 does, and run a linear layer on the result using nothing
but additions and subtractions of inputs. The quantizer is one line; what decides the
codes is which statistic sets the scale, and what happens to a weight that lies exactly
on the border between two codes.

## Specification

```python
def ternary_quantize(w): ...  # -> (t, scale)
def ternary_matmul(x, t, scale): ...  # -> y
```

**`ternary_quantize(w)`**. `w` is the weight of a linear layer, shape `(out, in)`. The
scale is the mean absolute value of **all** its elements, zeros included (the
"absmean"):

$$
\gamma = \frac{1}{\text{out} \cdot \text{in}} \sum_{i,j} |w_{ij}|,
\qquad
\text{scale} = \begin{cases} \gamma & \gamma > 0, \\ 1 & \gamma = 0 \ \text{(every weight is 0)}, \end{cases}
$$

$$
t_{ij} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{w_{ij}}{\text{scale}}\Big),\ -1,\ 1\Big).
$$

It returns the pair `(t, scale)`: `t` as an **int8** tensor of shape `(out, in)` whose
values are $-1$, $0$, or $1$, and `scale` as a 0-d float32 tensor. The matrix that the
pair stands for is $\hat W = \text{scale} \cdot t$.

**Rounding.** $\operatorname{round}$ is rounding to the nearest integer, and a value
exactly halfway between two integers goes to the **even** one, which is what `round`
does in PyTorch, JAX, and MLX. So $0.5$ and $-0.5$ round to $0$, and $1.5$ rounds to $2$,
which the clamp brings back to $1$. In terms of the weights:

$$
t_{ij} = \begin{cases}
\ \ 1 & w_{ij} > \text{scale} / 2, \\
\ \ 0 & |w_{ij}| \le \text{scale} / 2, \\
-1 & w_{ij} < -\text{scale} / 2 .
\end{cases}
$$

Some tests place weights **exactly** on the border $|w_{ij}| = \text{scale} / 2$. In
those tests the weights are small multiples of a power of two, so the mean and the
quotient are exact in float32 and the border case is well defined: it is code $0$. In
every other test, no value of $|w_{ij}| / \text{scale}$ is within $0.05$ of $0.5$, so a
one-bit difference in a float32 mean cannot change a code.

**`ternary_matmul(x, t, scale)`** applies the layer $y = x \hat W^{\top}$. Because
every weight is $-1$, $0$, or $+1$ times one scale, each output is a sum of some
inputs minus a sum of others, scaled once:

$$
y_i = \text{scale} \cdot \Big( \sum_{j \,:\, t_{ij} = 1} x_j \;-\; \sum_{j \,:\, t_{ij} = -1} x_j \Big).
$$

Inputs whose weight is $0$ contribute nothing.

- `x`: float32 activations of shape `(..., in)` with one to three dimensions. The last
  axis is the input axis; all other axes are batch axes.
- `t`: an int8 tensor of shape `(out, in)` with values in $\{-1, 0, 1\}$.
- `scale`: a 0-d float32 tensor, positive.
- Returns a float32 tensor of shape `(..., out)`.

**The rule of the exercise.** Compute the two sums by *selecting* inputs, not by
multiplying inputs with weights: no `matmul`, `einsum`, or elementwise product of `x`
with `t`. The only multiplication is the final one by `scale`.

> [!NOTE]
> The judge compares outputs, and a matrix product gives the same numbers, so it cannot
> see which one you wrote. The rule is the point of the layer all the same: a ternary
> kernel has no multiplier in its inner loop.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)` | float32 |
| `t` | `(out, in)` | int8 |
| `scale` | `()` | float32 |
| `x` | `(..., in)`, one to three dimensions | float32 |
| `y` | `(..., out)` | float32 |

## Example

```python
w = [
    [0.9, -0.2, 0.05, -1.3],
    [0.4, 0.0, -0.6, 0.6],
]

t, scale = ternary_quantize(w)
# scale = 0.50625
# t = [[1, 0,  0, -1],
#      [1, 0, -1,  1]]

ternary_matmul([1.0, 2.0, 3.0, 4.0], t, scale)  # [-1.51875, 1.0125]
```

The absolute values sum to $4.05$ over 8 elements, so $\text{scale} = 0.50625$, and a
weight becomes $\pm 1$ when its magnitude exceeds $\text{scale} / 2 = 0.253125$:

| $w$ | $0.9$ | $-0.2$ | $0.05$ | $-1.3$ | $0.4$ | $0.0$ | $-0.6$ | $0.6$ |
|---|---|---|---|---|---|---|---|---|
| $w / \text{scale}$ | $1.78$ | $-0.40$ | $0.10$ | $-2.57$ | $0.79$ | $0$ | $-1.19$ | $1.19$ |
| $t$ | $1$ | $0$ | $0$ | $-1$ | $1$ | $0$ | $-1$ | $1$ |

For $x = [1, 2, 3, 4]$, output 0 adds $x_0$ and subtracts $x_3$:
$0.50625 \cdot (1 - 4) = -1.51875$. Output 1 adds $x_0$ and $x_3$ and subtracts $x_2$:
$0.50625 \cdot (1 + 4 - 3) = 1.0125$.

A border case: for `w = [[0.5, -0.5, 2.0, -1.0], [0.0, 1.0, 1.5, 1.5]]` the absolute
values sum to $8$, so the scale is exactly $1$, and `t` is
`[[0, 0, 1, -1], [0, 1, 1, 1]]`. The weights $\pm 0.5$ are exactly halfway and become
$0$; $1.5$ rounds to $2$ and is clamped to $1$.

## Constraints

- $1 \le \text{out}, \text{in}$; `w` has at most 100,000 elements, each finite with
  $|w_{ij}| \le 10^4$. A matrix that is not all zeros has $\gamma \ge 10^{-6}$.
- `t` is compared **exactly** and must have dtype int8. `scale` is compared to a
  relative tolerance of `1e-4`.
- The result of `ternary_matmul` is compared to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. `x` has at most 10,000 elements with $|x_j| \le 100$,
  and the number of elements of `x` times `out` is at most 1,000,000.
- No Python loops over elements, rows, or batch entries are needed.

## Why this matters

A ternary weight carries $\log_2 3 \approx 1.58$ bits, a tenth of a float16 weight, and
a ternary layer needs no multiplier: the BitNet b1.58 paper builds its case on matrix
products that consist of additions. This problem is the inference side of that design,
the function an engine applies when it loads such a model and the kernel it runs
afterwards. The models are trained with this exact quantizer in the forward pass, so
an engine that deviates from it in any detail runs weights the model was never
trained with. The details are the ones above: the scale is the mean magnitude, not the
maximum (with the maximum, nearly every weight rounds to 0); the mean counts the zeros;
there is one scale for the matrix; and the zero code is not a rounding accident but the
third value, which lets the model switch an input off. A kernel that treats $0$ as
$+1$ computes the earlier, binary BitNet instead, and produces fluent nonsense.
