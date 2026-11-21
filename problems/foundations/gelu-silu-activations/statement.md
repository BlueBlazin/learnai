Implement the two activation functions of modern language models: GELU, in both forms
that checkpoints are trained with, and SiLU. The formulas are short. The work is in
telling the two GELUs apart, getting the constants right, and returning a number
instead of NaN when an activation is large.

![This problem is the activation inside the MLP of each block.](assets/map.svg)

## Specification

```python
def gelu(x, approximate="none"): ...
def silu(x): ...
```

Both functions are applied elementwise.

**GELU** (Hendrycks and Gimpel, 2016) multiplies its input by the standard normal
cumulative distribution function $\Phi$:

$$
\operatorname{gelu}(x) = x \, \Phi(x) = \frac{x}{2} \left( 1 + \operatorname{erf}\!\left( \frac{x}{\sqrt{2}} \right) \right) .
$$

The same paper gives an approximation that avoids the error function:

$$
\operatorname{gelu}_{\tanh}(x) = \frac{x}{2} \left( 1 + \tanh\!\left( \sqrt{\frac{2}{\pi}} \, \big( x + 0.044715 \, x^3 \big) \right) \right) .
$$

- `approximate="none"` (the default) returns the exact form, computed with your
  framework's `erf`.
- `approximate="tanh"` returns the approximation.

These are two different functions: they differ by up to $4.7 \times 10^{-4}$, and a
checkpoint is trained with one of them. No other value of `approximate` is passed.

**SiLU** (also called swish) multiplies its input by the logistic sigmoid:

$$
\operatorname{silu}(x) = x \, \sigma(x), \qquad \sigma(x) = \frac{1}{1 + e^{-x}} .
$$

- `x`: a float32 tensor of any shape with one to four dimensions.
- Returns a float32 tensor of the same shape.

Every result must be a finite number for every input in the range given under
Constraints, including values such as `1e4` and `-1e4`.

Write the arithmetic yourself. Do not call a built-in GELU or SiLU (`gelu`, `silu`,
`swish`, and their layer classes), a built-in sigmoid (`sigmoid`, `expit`,
`log_sigmoid`), or a built-in normal CDF (`ndtr`, `norm.cdf`). The elementwise functions
`erf`, `tanh`, and `exp` are what you need.

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

For $x = 1$: $\Phi(1) = 0.841345$, so the exact GELU is $0.841345$. The tanh form
computes $\sqrt{2/\pi}\,(1 + 0.044715) = 0.833562$, $\tanh(0.833562) = 0.682384$, and
$0.5 \times 1.682384 = 0.841192$: close, and not the same. At $x = -3$ the two forms
differ by 10 percent of the value.

For SiLU, $\sigma(1) = 0.731059$ and $\sigma(-3) = 0.047426$, so
$\operatorname{silu}(-3) = -0.142278$.

Large inputs have simple answers, since $\Phi$ and $\sigma$ saturate at 0 and 1:

```python
silu([-100.0, 100.0])  # [-0.0, 100.0]
```

A sigmoid written as $e^{x} / (1 + e^{x})$ returns NaN for the second entry, because
$e^{100}$ overflows float32 (anything above $e^{88.72}$ does) and `inf / inf` is NaN.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN or an infinity is always wrong. A result of
  `-0.0` counts as `0.0`.
- Every entry of `x` is finite with $|x| \le 10^4$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32.

## Why this matters

The activation is part of the trained weights' contract, and a config file names it
with one word. GPT-2 and Gemma were trained with the tanh form, other models with the erf form, and
the feed-forward layers of Llama, Qwen, and Mistral use SiLU. An engine that
loads a checkpoint with the wrong GELU does not crash. Every activation is off in the
fourth decimal place, the error compounds through dozens of layers, and the logits drift
far enough to change which token wins a close call, which shows up as a model that is
slightly worse than the reference for no visible reason. The overflow matters for the
same quiet reason: one NaN from one large activation spreads to every position that
attends to it.
