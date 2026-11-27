Choose how much of a tensor's value range to keep before quantizing it. Given calibration samples and a list of candidate clipping fractions, quantize the samples at each candidate range and return the fraction that gives the lowest mean squared error.

## Specification

```python
def best_clip(samples, bits, candidates): ...  # -> the winning candidate, a 0-d float32
```

All elements of `samples` are pooled into a single flat calibration set so that one quantization range applies to the entire tensor. Let $n$ be the total number of elements in `samples`, and let $m = \max |x|$ be the largest magnitude across all elements $x$ of `samples`.

Quantization uses a signed symmetric integer grid with bit width `bits`. The largest representable integer code is $q_{\max} = 2^{\text{bits} - 1} - 1$, giving $2 q_{\max} + 1$ integer levels from $-q_{\max}$ to $q_{\max}$. For each candidate clipping fraction $c$ in `candidates`, set the clipping range to $[-c \cdot m, c \cdot m]$ and compute the step size `scale`:

$$
\text{limit} = c \cdot m, \qquad \text{scale} = \frac{\text{limit}}{q_{\max}}.
$$

Quantize every element $x$ by dividing by `scale`, rounding to the nearest integer, clamping to $[-q_{\max}, q_{\max}]$, and multiplying back by `scale` to obtain the dequantized value $\hat x$. Then compute the mean squared error $\text{MSE}(c)$ over all $n$ elements:

$$
\hat x = \text{scale} \cdot \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{x}{\text{scale}}\Big),\ -q_{\max},\ q_{\max}\Big),
\qquad
\text{MSE}(c) = \frac{1}{n} \sum_{x} (\hat x - x)^2 .
$$

Values outside $[-\text{limit}, \text{limit}]$ saturate at $\pm \text{limit}$ and incur clipping error. Values inside the range round to the nearest grid point and incur rounding error.

![Lower the clipping fraction: the levels close in on the bulk and rounding error falls, while the tinted tails saturate and clipping error rises.](interactive:quant-grid)

`best_clip` returns the candidate $c$ with the lowest $\text{MSE}(c)$. If multiple candidates tie for the lowest MSE, return the **largest** of them (the widest range).

- `samples`: every element is treated as one calibration value regardless of shape. At least one element is nonzero.
- `bits`: a Python int with $2 \le \text{bits} \le 8$. For example, 4 bits give codes from $-7$ to $7$, and 2 bits give $-1, 0, 1$.
- `candidates`: $K \ge 1$ distinct fractions in $(0, 1]$ in **any order**. A fraction of $1$ means no clipping, and $1$ is not required to appear in the list.
- Return value: one of the values from `candidates`.

**Rounding.** $\operatorname{round}$ rounds to the nearest integer with ties to even, matching the default `round` behavior in PyTorch, JAX, and MLX. A value halfway between two grid points has the same squared error either way.

## Shapes

| name | shape | dtype |
|---|---|---|
| `samples` | `(N,)` or `(N, D)` | float32 |
| `candidates` | `(K,)` | float32 |
| result | `()` | float32 |

## Example

```python
samples = [0.8, -0.9, 1.1, -1.2, 0.6, 2.0]
candidates = [1.0, 0.5, 0.25]

best_clip(samples, bits=2, candidates=candidates)  # 0.5
```

Here the largest magnitude is $m = 2.0$, and `bits = 2` gives $q_{\max} = 1$. Each candidate therefore produces a three-point grid at $-\text{limit}$, $0$, and $\text{limit}$.

| $c$ | limit | $\hat x$ for each sample | squared errors | MSE |
|---|---|---|---|---|
| $1.0$ | $2.0$ | $0,\ 0,\ 2,\ -2,\ 0,\ 2$ | $0.64,\ 0.81,\ 0.81,\ 0.64,\ 0.36,\ 0$ | $0.5433$ |
| $0.5$ | $1.0$ | $1,\ -1,\ 1,\ -1,\ 1,\ 1$ | $0.04,\ 0.01,\ 0.01,\ 0.04,\ 0.16,\ 1$ | $0.21$ |
| $0.25$ | $0.5$ | $0.5,\ -0.5,\ 0.5,\ -0.5,\ 0.5,\ 0.5$ | $0.09,\ 0.16,\ 0.36,\ 0.49,\ 0.01,\ 2.25$ | $0.56$ |

Without clipping ($c = 1.0$), the single outlier $2.0$ sets the grid, so the five values near $\pm 1$ round to $0$ or $\pm 2$ with large error. At $c = 0.5$, the outlier is clipped to $1.0$ for a squared error of $1$, while the other five values land close to $\pm 1$, cutting total MSE by more than half. At $c = 0.25$, the range is too narrow for every sample. The winning candidate is $0.5$.

## Constraints

- `samples` has at most 4096 elements with $|x| \le 1000$, and $K \le 16$.
- The result is compared with the reference and must equal one of the candidates.
- Float32 rounding never decides a test. In every test, either the lowest MSE is at least 2 percent below the MSE of every other candidate, or the test checks tie-breaking. Tie tests use 2 bits, integer-valued samples whose largest magnitude is a power of two, and candidates that are multiples of $1/8$, so every limit, quantized value, and squared error is a small exact value.
- No Python loops over samples are needed. A loop over the candidates is acceptable.

## Why this matters

Min-max quantization lets the single largest value determine the step size for the entire tensor. Real neural network weights and activations are bell-shaped with long tails, so a 4-bit min-max grid wastes most of its 15 levels on sparse extreme values.

Production quantization pipelines therefore calibrate a tighter clipping range on sample data. GPTQ shrinks the range in 1 percent steps and keeps the best MSE, AWQ searches over clipping fractions after per-channel scaling, and static activation quantizers pick ranges using MSE, percentile, or entropy calibrators. Small bugs in this search fail silently. Measuring only clipping error disables clipping completely, omitting the clamp makes the search pick the smallest candidate, and computing a separate range per sample calibrates a scale that a deployed single-scale tensor cannot use.
