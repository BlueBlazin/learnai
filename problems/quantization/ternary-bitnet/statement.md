Implement ternary weight quantization and a multiplier-free linear layer following BitNet b1.58. You will quantize a weight matrix to the three values $-1$, $0$, and $+1$ using a single scale for the whole matrix, then compute the linear layer output using only additions and subtractions of inputs.

## Specification

```python
def ternary_quantize(w): ...  # -> (t, scale)
def ternary_matmul(x, t, scale): ...  # -> y
```

### `ternary_quantize(w)`

Given a float32 weight matrix `w` of shape `(out, in)`, compute the mean absolute value $\gamma$ over all $\text{out} \cdot \text{in}$ elements, including zeros:

$$
\gamma = \frac{1}{\text{out} \cdot \text{in}} \sum_{i,j} |w_{ij}|,
\qquad
\text{scale} = \begin{cases} \gamma & \text{if } \gamma > 0, \\ 1 & \text{if } \gamma = 0 \text{ (every weight is 0)}. \end{cases}
$$

Next, divide each weight by `scale`, round to the nearest integer, and clamp to $[-1, 1]$ to obtain the ternary codes $t$:

$$
t_{ij} = \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{w_{ij}}{\text{scale}}\Big),\ -1,\ 1\Big).
$$

Return the pair `(t, scale)`, where `t` has dtype `int8` with values in $\{-1, 0, 1\}$, and `scale` is a 0-d float32 tensor. Together they represent the quantized weight matrix $\hat W = \text{scale} \cdot t$.

**Rounding.** Here $\operatorname{round}$ rounds to the nearest integer with ties going to the even integer, which matches `round` in PyTorch, JAX, and MLX. For example, $0.5$ and $-0.5$ round to $0$, while $1.5$ rounds to $2$ and is then clamped to $1$. Equivalently, each code is determined by comparing $w_{ij}$ against $\pm \text{scale} / 2$:

$$
t_{ij} = \begin{cases}
\ \ 1 & w_{ij} > \text{scale} / 2, \\
\ \ 0 & |w_{ij}| \le \text{scale} / 2, \\
-1 & w_{ij} < -\text{scale} / 2 .
\end{cases}
$$

Some tests place weights exactly on the border $|w_{ij}| = \text{scale} / 2$. In those tests, the weights are small multiples of a power of two so the mean and quotient are exact in float32, and the border weight maps to $0$. In all other tests, no $|w_{ij}| / \text{scale}$ lies within $0.05$ of $0.5$, so float32 rounding differences in the mean cannot change any code.

### `ternary_matmul(x, t, scale)`

Apply the quantized linear layer $y = x \hat W^{\top}$ to float32 activations `x` of shape `(..., in)` with one to three dimensions, where the last axis is the input axis and any leading axes are batch axes. The arguments `t` (with values in $\{-1, 0, 1\}$) and `scale` (positive) are the outputs of `ternary_quantize`.

Because every quantized weight is $-1$, $0$, or $+1$ multiplied by the same `scale`, each output entry $y_i$ equals the sum of inputs where $t_{ij} = 1$ minus the sum of inputs where $t_{ij} = -1$, scaled once at the end:

$$
y_i = \text{scale} \cdot \Big( \sum_{j \,:\, t_{ij} = 1} x_j \;-\; \sum_{j \,:\, t_{ij} = -1} x_j \Big).
$$

Inputs whose weight code is $0$ contribute nothing.

**Rule of the exercise.** Compute the two sums by selecting inputs rather than multiplying inputs with weights. Do not use `matmul`, `einsum`, or an elementwise product of `x` with `t`. The only multiplication should be the final one by `scale`.

> [!NOTE]
> Because the judge compares output values and a matrix product gives the same numbers, it cannot enforce this rule mechanically. Follow it anyway, since avoiding input-weight multiplication is the core idea of a ternary kernel.

## Shapes

| name | shape | dtype |
|---|---|---|
| `w` | `(out, in)` | float32 |
| `t` | `(out, in)` | int8 |
| `scale` | `()` | float32 |
| `x` | `(..., in)`, 1 to 3 dimensions | float32 |
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

The absolute values of `w` sum to $4.05$ across 8 elements, so $\text{scale} = 4.05 / 8 = 0.50625$. A weight becomes $\pm 1$ when its magnitude exceeds $\text{scale} / 2 = 0.253125$:

| $w$ | $0.9$ | $-0.2$ | $0.05$ | $-1.3$ | $0.4$ | $0.0$ | $-0.6$ | $0.6$ |
|---|---|---|---|---|---|---|---|---|
| $w / \text{scale}$ | $1.78$ | $-0.40$ | $0.10$ | $-2.57$ | $0.79$ | $0$ | $-1.19$ | $1.19$ |
| $t$ | $1$ | $0$ | $0$ | $-1$ | $1$ | $0$ | $-1$ | $1$ |

For $x = [1, 2, 3, 4]$, output 0 adds $x_0$ and subtracts $x_3$, giving $0.50625 \cdot (1 - 4) = -1.51875$. Output 1 adds $x_0$ and $x_3$ and subtracts $x_2$, giving $0.50625 \cdot (1 + 4 - 3) = 1.0125$.

As a border case, consider `w = [[0.5, -0.5, 2.0, -1.0], [0.0, 1.0, 1.5, 1.5]]`. The absolute values sum to $8$, so `scale` is exactly $1$, and `t` is `[[0, 0, 1, -1], [0, 1, 1, 1]]`. The entries $\pm 0.5$ sit exactly halfway and round to $0$, while $1.5$ rounds to $2$ and is clamped to $1$.

## Constraints

- $1 \le \text{out}, \text{in}$. `w` has at most 100,000 elements, each finite with $|w_{ij}| \le 10^4$. Any matrix that is not all zeros has $\gamma \ge 10^{-6}$.
- `t` is compared **exactly** and must have dtype `int8`. `scale` is compared with a relative tolerance of `1e-4`.
- The output of `ternary_matmul` is compared with a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. `x` has at most 10,000 elements with $|x_j| \le 100$, and the number of elements in `x` times `out` is at most 1,000,000.
- No Python loops over elements, rows, or batch entries are needed.

## Why this matters

A ternary weight carries $\log_2 3 \approx 1.58$ bits of information, roughly a tenth of a float16 weight, and a ternary linear layer requires no multipliers in its inner loop. BitNet b1.58 trains models with this exact quantizer in the forward pass, so an inference engine must match every detail when loading and running the weights.

Small mismatches silently break the model. Using the maximum magnitude instead of the mean sends almost every weight to $0$. Excluding zeros from the mean shifts the threshold for all remaining weights. Treating $0$ as $+1$ removes the model's ability to zero out an input, turning ternary BitNet b1.58 into binary BitNet and producing fluent nonsense.
