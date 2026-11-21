Implement the two partial rotary embeddings that current open-weight models ship. Both
read the same head width, base, and `partial_rotary_factor` from a configuration, and
they disagree on which channels rotate, how they are paired, and at what frequencies.

## Specification

```python
def partial_rope(x, positions, base, factor): ...
def proportional_rope(x, positions, base, factor): ...
```

`x` has shape `(B, H, T, D)` with `D` even, and `positions[t]` is the absolute position
$p$ of the token at sequence index `t`. Both functions rotate pairs of channels
$(a, b)$ counterclockwise by the angle $p\,\theta_i$, as in the `rope` problem,

$$
\begin{bmatrix} a' \\ b' \end{bmatrix}
=
\begin{bmatrix}
\cos p\theta_i & -\sin p\theta_i \\
\sin p\theta_i & \phantom{-}\cos p\theta_i
\end{bmatrix}
\begin{bmatrix} a \\ b \end{bmatrix},
$$

write the rotated members back to the channels they came from, and return every other
channel unchanged. They differ in the rest:

| | `partial_rope` | `proportional_rope` |
|---|---|---|
| configuration | `rope_type: "default"` with a `partial_rotary_factor` | `rope_type: "proportional"` |
| number of rotated pairs | $r/2$, with $r = \texttt{int}(D \cdot \texttt{factor})$ | $n = \texttt{int}(\texttt{factor} \cdot D \mathbin{//} 2)$ |
| pair $i$ is $(a, b) =$ | $(x_i,\; x_{i + r/2})$, $0 \le i < r/2$ | $(x_i,\; x_{i + D/2})$, $0 \le i < n$ |
| frequency of pair $i$ | $\theta_i = \texttt{base}^{-2i/r}$ | $\theta_i = \texttt{base}^{-2i/D}$ |
| unchanged channels | $r, \dots, D - 1$ | $n, \dots, D/2 - 1$ and $D/2 + n, \dots, D - 1$ |

`partial_rope` is a complete half-split RoPE of width $r$ on the first $r$ channels.
`proportional_rope` is the half-split RoPE of the whole head in which only the first
$n$ pairs turn; the other $D/2 - n$ pairs have frequency zero.

Both counts are the Python expressions shown, evaluated on floats:
`int(D * factor)`, and `int(factor * D // 2)` with the product first and the floor
division by 2 second.

![A head of width 16 with `factor = 0.5`. `partial_rope` rotates channels 0 to 7 and pairs channel $i$ with $i + 4$; `proportional_rope` rotates channels 0 to 3 and 8 to 11 and pairs channel $i$ with $i + 8$, and its frequency exponents have different denominators.](assets/channel-map.svg)

- `x`: float32, shape `(B, H, T, D)`, `D` even.
- `positions`: integer tensor, shape `(T,)`, values $\ge 0$; not necessarily starting
  at 0, contiguous, sorted, or distinct.
- `base`: a Python float greater than 1 (`rope_theta` in configurations).
- `factor`: a Python float, a multiple of $1/8$ with $0 < \texttt{factor} \le 1$, so
  $D \cdot \texttt{factor}$ is computed without rounding. For `partial_rope` it is an
  even integer. For `proportional_rope`, $\texttt{factor} \cdot D / 2$ need not be an
  integer and is floored.
- Both return float32, shape `(B, H, T, D)`.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(B, H, T, D)`, `D` even | float32 |
| `positions` | `(T,)` | integer |
| result of either function | `(B, H, T, D)` | float32 |

## Example

One vector of eight ones at position 1, with `base = 100` and `factor = 0.5`:

```python
x = [[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]]  # shape (1, 1, 1, 8)

partial_rope(x, [1], 100.0, 0.5)
# [[[[-0.301169, 0.895171, 1.381773, 1.094838, 1.0, 1.0, 1.0, 1.0]]]]

proportional_rope(x, [1], 100.0, 0.5)
# [[[[-0.301169, 0.639432, 1.0, 1.0, 1.381773, 1.261399, 1.0, 1.0]]]]
```

`partial_rope`: $r = 4$ and $\theta = [100^{0}, 100^{-2/4}] = [1, 0.1]$. Pair 0 is
channels $(0, 2) = (1, 1)$, turned by 1 radian to
$(\cos 1 - \sin 1,\ \sin 1 + \cos 1) = (-0.3012, 1.3818)$. Pair 1 is channels $(1, 3)$,
turned by 0.1 radian to $(0.8952, 1.0948)$.

`proportional_rope`: $n = \texttt{int}(0.5 \cdot 8 \mathbin{//} 2) = 2$ and
$\theta = [100^{0}, 100^{-2/8}] = [1, 0.3162]$. Pair 0 is channels $(0, 4)$, turned by
1 radian. Pair 1 is channels $(1, 5)$, turned by 0.3162 radian to $(0.6394, 1.2614)$.

## Constraints

- $1 \le B, H, T \le 8$; $D$ is even, $4 \le D \le 512$; $10 \le \texttt{base} \le
  10^7$; $0 \le$ `positions[t]` $< 2048$.
- Results are compared with the reference and, for the parameters of Qwen3.5 and
  Gemma 4, with arrays computed by the rotary code of Hugging Face Transformers. When
  every position is below 64: relative tolerance `1e-4`, absolute tolerance `1e-4`.
  With larger positions: relative tolerance `1e-3`, absolute tolerance `5e-3`, because
  equally reasonable ways of computing the angle $p\,\theta_i$ in float32 differ by
  $p \cdot 10^{-7}$ radians.
- Dot products of two rotated vectors must depend on their positions only through the
  difference (tolerance `1e-3`).
- No Python loops over positions or pairs are needed.

## Why this matters

Qwen3.5 (`rope_type: "default"`, factor 0.25, head width 256) rotates its first 64
channels with the frequencies of a 64-wide head. Gemma 4's full-attention layers
(`rope_type: "proportional"`, factor 0.25, head width 512) rotate the first 64 of 256
pairs with the frequencies of the 512-wide head. An engine that implements one
convention and loads the other model raises no error: the shapes agree, the first pair
gets the right angle, and attention runs with the wrong notion of distance.
