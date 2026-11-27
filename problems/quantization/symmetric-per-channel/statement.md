Implement symmetric per-channel quantization and dequantization for a 2D weight matrix. Each output row gets its own scale factor and maps weights to signed integer codes centered on zero, with no zero-point.

## Specification

```python
def quantize_weight(w, bits): ...  # -> (q, scale)
def dequantize(q, scale): ...  # -> float32 weights
```

`w` is the weight matrix of a linear layer with shape `(out, in)`, where row $i$ holds the weights for output channel $i$. A symmetric grid represents a real number $r$ by an integer code $q$ using $r = \text{scale} \cdot q$.

For a given bit width `bits`, the valid integer codes are

$$
q \in \{-q_{\max}, \dots, q_{\max}\}, \qquad q_{\max} = 2^{\,\text{bits} - 1} - 1.
$$

For example, 8 bits use $[-127, 127]$ and 4 bits use $[-7, 7]$. The most negative two's-complement value, $-2^{\,\text{bits} - 1}$, is never used, leaving the same number of codes on both sides of zero.

**`quantize_weight(w, bits)`** computes the maximum magnitude $m_i$ and scale $\text{scale}_i$ independently for each row $i$:

$$
m_i = \max_j |w_{ij}|, \qquad
\text{scale}_i = \begin{cases}
m_i / q_{\max} & \text{if } m_i > 0, \\
1 & \text{if } m_i = 0 \ \text{(all-zero row)},
\end{cases}
$$

$$
q_{ij} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{w_{ij}}{\text{scale}_i}\Big),\ -q_{\max},\ q_{\max}\Big).
$$

Return `(q, scale)`. `q` must have dtype `int8` regardless of `bits`, and `scale` has one float32 entry per output row. In any nonzero row, the largest-magnitude weight maps to $+q_{\max}$ or $-q_{\max}$ according to its sign. An all-zero row receives scale `1.0` and all-zero codes, with no `NaN` or infinite values.

**`dequantize(q, scale)`** reconstructs the float32 weight matrix by multiplying each row $i$ of `q` by its row scale $\text{scale}_i$:

$$
\hat{w}_{ij} = \text{scale}_i \cdot q_{ij}.
$$

**Rounding.** $\operatorname{round}$ rounds to the nearest integer, breaking halfway ties to the nearest even integer. This matches `round` in PyTorch, JAX, and MLX. The tests contain no value of $w_{ij} / \text{scale}_i$ within $0.05$ of a tie, so a one-bit float32 difference cannot change a code. Truncation is not rounding and will fail.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)` | float32 |
| `bits` | Python `int`, $2 \le \text{bits} \le 8$ | |
| `q` | `(out, in)` | `int8` |
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

Row 0 has $m_0 = 1.0$, giving $\text{scale}_0 = 1 / 127$ and codes $\operatorname{round}(127 \cdot [0.4, -1.0, 0.25, 0.8]) = \operatorname{round}([50.8, -127, 31.75, 101.6])$. Row 1 is roughly 30 times smaller ($m_1 = 0.03$) and gets its own scale $0.03 / 127$, so its codes $\operatorname{round}([84.67, 42.33, -127, 0])$ still use the full range. Under a single per-tensor scale of $1 / 127$, row 1 would have collapsed to `[3, 1, -4, 0]`. Row 2 is all zeros, so it gets scale `1.0` and zero codes.

With `bits = 4`, $q_{\max} = 7$. The scales are `[0.142857, 0.00428571, 1.0]` and the codes are `[[3, -7, 2, 6], [5, 2, -7, 0], [0, 0, 0, 0]]`.

## Constraints

- $1 \le \text{out}, \text{in}$ and `w` has at most 100,000 elements. Every weight is finite with $|w_{ij}| \le 10^4$, and every nonzero row has $m_i \ge 10^{-6}$.
- The codes are compared **exactly** and must have dtype `int8`. `scale` and the dequantized weights are compared to a relative tolerance of `1e-4` and an absolute tolerance of `1e-6`.
- After the round trip, every weight is within half a step of its row: $|\hat{w}_{ij} - w_{ij}| \le \text{scale}_i / 2$.
- No Python loop over rows or elements is needed.

## Why this matters

Symmetric per-channel quantization is the default weight format across almost every integer inference stack. Fixing the zero-point to 0 avoids extra cross terms during integer matrix multiplication, so the product of codes equals the product of values up to a single scale factor. Using one scale per output row preserves precision when rows differ in magnitude by an order of magnitude or more, while adding almost no runtime cost because each row scale is applied once after integer accumulation.

Axis bugs are the most common silent failure mode here. Computing scales per column or broadcasting `(out,)` against the trailing axis produces a layer that runs without error on square weight matrices while quietly destroying model accuracy.
