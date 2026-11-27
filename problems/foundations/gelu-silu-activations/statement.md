Implement `gelu` (in both its exact and `tanh`-approximated forms) and `silu`, the two activation functions used in modern language models. Your implementations must match the exact formulas and remain numerically finite for large inputs.

![This problem is the activation inside the MLP of each block.](assets/map.svg)

## Specification

```python
def gelu(x, approximate="none"): ...
def silu(x): ...
```

Both functions operate elementwise on a float32 tensor `x` and return a float32 tensor of the same shape.

**GELU** (Hendrycks and Gimpel, 2016) multiplies each input $x$ by $\Phi(x)$, the cumulative distribution function (CDF) of the standard normal distribution. Because $\Phi(x)$ transitions smoothly from 0 for large negative values to 1 for large positive values, it acts as a smooth gate:

$$
\operatorname{gelu}(x) = x \, \Phi(x) = \frac{x}{2} \left( 1 + \operatorname{erf}\!\left( \frac{x}{\sqrt{2}} \right) \right) .
$$

The same paper defines a `tanh`-based approximation that avoids calling `erf`:

$$
\operatorname{gelu}_{\tanh}(x) = \frac{x}{2} \left( 1 + \tanh\!\left( \sqrt{\frac{2}{\pi}} \, \big( x + 0.044715 \, x^3 \big) \right) \right) .
$$

- `approximate="none"` (the default) returns the exact form, computed with your framework's `erf`.
- `approximate="tanh"` returns the `tanh` approximation.

These are two distinct functions that differ by up to $4.7 \times 10^{-4}$, and trained checkpoints depend on the specific form they were trained with. No value of `approximate` other than `"none"` or `"tanh"` is passed.

**SiLU** (also called swish) multiplies each input $x$ by the logistic sigmoid $\sigma(x)$:

$$
\operatorname{silu}(x) = x \, \sigma(x), \qquad \sigma(x) = \frac{1}{1 + e^{-x}} .
$$

Every output element must be finite for all inputs in the range given under Constraints, including large values such as `1e4` and `-1e4`.

Write the arithmetic yourself using elementwise primitives (`erf`, `tanh`, and `exp`). Do not call built-in GELU or SiLU functions or layer classes (`gelu`, `silu`, `swish`), built-in sigmoids (`sigmoid`, `expit`, `log_sigmoid`), or built-in normal CDFs (`ndtr`, `norm.cdf`).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(d_1, ..., d_k)`, $1 \le k \le 4$ | float32 |
| result | `(d_1, ..., d_k)` | float32 |

## Example

```python
x = [-3.0, -1.0, 0.0, 1.0, 3.0]

gelu(x)                      # [-0.004050, -0.158655, 0.0, 0.841345, 2.995950]
gelu(x, approximate="tanh")  # [-0.003637, -0.158808, 0.0, 0.841192, 2.996363]
silu(x)                      # [-0.142278, -0.268941, 0.0, 0.731059, 2.857723]
```

At $x = 1$, $\Phi(1) = 0.841345$, so exact GELU returns `0.841345`. The `tanh` form computes $\sqrt{2/\pi}\,(1 + 0.044715) = 0.833562$, then $\tanh(0.833562) = 0.682384$, and finally $0.5 \times 1.682384 = 0.841192$. The two values are close, but not identical. At $x = -3$, the two forms differ by about 10 percent.

For SiLU, $\sigma(1) = 0.731059$ and $\sigma(-3) = 0.047426$, which gives $\operatorname{silu}(-3) = -0.142278$.

For large inputs, both $\Phi$ and $\sigma$ saturate to 0 or 1:

```python
silu([-100.0, 100.0])  # [-0.0, 100.0]
```

Writing the sigmoid as $e^{x} / (1 + e^{x})$ returns NaN for the second entry. In float32, $e^{x}$ overflows above $x = 88.72$, so $e^{100}$ becomes `inf` and `inf / inf` evaluates to NaN.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an absolute tolerance of `1e-5`. Any NaN or infinity fails. A result of `-0.0` counts as `0.0`.
- Every entry of `x` is finite with $|x| \le 10^4$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32.

## Why this matters

The activation function is part of a trained checkpoint's contract, and a model config names it with a single string. GPT-2 and Gemma were trained with the `tanh` form of GELU, other models use the exact `erf` form, and the feed-forward layers of Llama, Qwen, and Mistral use SiLU.

An inference engine that loads a checkpoint with the wrong GELU variant does not crash. Every activation shifts in the fourth decimal place, the error compounds through dozens of layers, and the final logits drift far enough to flip close token predictions. Overflow fails just as quietly, because a single NaN from one large activation propagates to every position that attends to it.
