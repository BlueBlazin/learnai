Choose how much of a tensor's range to keep before quantizing it. Given calibration
samples and a list of candidate clipping fractions, quantize the samples once per
candidate and return the fraction with the lowest mean squared error. A smaller range
makes the grid finer for the bulk of the values and destroys the outliers; the error
you minimize has to count both.

## Specification

```python
def best_clip(samples, bits, candidates): ...  # -> the winning candidate, a 0-d float32
```

Let $m = \max |x|$ over **all** elements $x$ of `samples`: the calibration set is
pooled, and there is one range for all of it. Let
$q_{\max} = 2^{\text{bits} - 1} - 1$. For a candidate fraction $c$, the samples are
quantized on a symmetric grid that reaches $\pm c \cdot m$:

$$
\text{limit} = c \cdot m, \qquad \text{scale} = \frac{\text{limit}}{q_{\max}},
$$

$$
\hat x = \text{scale} \cdot \operatorname{clamp}\!\Big(\operatorname{round}\!\Big(\frac{x}{\text{scale}}\Big),\ -q_{\max},\ q_{\max}\Big),
\qquad
\text{MSE}(c) = \frac{1}{N} \sum_{x} (\hat x - x)^2 ,
$$

where the sum runs over all $N$ elements of `samples`. Values beyond the limit
saturate at $\pm \text{limit}$; values inside it are rounded to the nearest of the
$2 q_{\max} + 1$ grid points.

![Lower the clipping fraction: the levels close in on the bulk and rounding error falls, while the tinted tails saturate and clipping error rises.](interactive:quant-grid)

`best_clip` returns the candidate with the lowest $\text{MSE}(c)$. If several
candidates reach the lowest MSE, it returns the **largest** of them (the larger range).

- `samples`: a float32 tensor of shape `(N,)` or `(N, D)`. The shape carries no
  meaning: every element is one calibration value. At least one element is nonzero.
- `bits`: a Python int, $2 \le \text{bits} \le 8$. The codes are signed and symmetric,
  $-q_{\max}, \dots, q_{\max}$ (at 4 bits: $-7$ to $7$; at 2 bits: $-1, 0, 1$).
- `candidates`: a float32 tensor of shape `(K,)` with $K \ge 1$ distinct fractions in
  $(0, 1]$, in **any order**. The fraction 1 means no clipping. It is not necessarily
  in the list.
- Returns a 0-d float32 tensor holding one of the values of `candidates`.

**Rounding.** $\operatorname{round}$ is rounding to the nearest integer, ties to even,
as `round` does in PyTorch, JAX, and MLX. A value exactly halfway between two grid
points has the same squared error whichever way it goes.

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

The largest magnitude is $m = 2.0$, and 2 bits give $q_{\max} = 1$: the grid of each
candidate has just three points, $-\text{limit}$, $0$, and $\text{limit}$.

| $c$ | limit | $\hat x$ for each sample | squared errors | MSE |
|---|---|---|---|---|
| $1.0$ | $2.0$ | $0,\ 0,\ 2,\ -2,\ 0,\ 2$ | $0.64,\ 0.81,\ 0.81,\ 0.64,\ 0.36,\ 0$ | $0.5433$ |
| $0.5$ | $1.0$ | $1,\ -1,\ 1,\ -1,\ 1,\ 1$ | $0.04,\ 0.01,\ 0.01,\ 0.04,\ 0.16,\ 1$ | $0.21$ |
| $0.25$ | $0.5$ | $0.5,\ -0.5,\ 0.5,\ -0.5,\ 0.5,\ 0.5$ | $0.09,\ 0.16,\ 0.36,\ 0.49,\ 0.01,\ 2.25$ | $0.56$ |

Without clipping ($c = 1$), the grid is set by the single value $2.0$, and the five
values near $\pm 1$ are rounded to $0$ or to $\pm 2$, far from where they are. At
$c = 0.5$ the outlier is clipped to $1.0$ and pays a squared error of $1$, but the
other five now sit next to a grid point, and the total drops by more than half. At
$c = 0.25$ the range is too small for everything. The answer is $0.5$.

## Constraints

- `samples` has at most 4096 elements with $|x| \le 1000$, and $K \le 16$.
- The result is compared with the reference and must equal one of the candidates.
- Float32 rounding never decides a test. In every test, either the lowest MSE is at
  least 2 percent below the MSE of every other candidate, or the test is about ties.
  Tie tests use 2 bits, integer-valued samples whose largest magnitude is a power of
  two, and candidates that are multiples of $1/8$, so every limit, quantized value,
  and squared error is a small integer and tied candidates are tied exactly.
- No Python loops over samples are needed. A loop over the candidates is acceptable.

## Why this matters

Min-max quantization lets the single largest value decide the step for everyone. Real
weights and activations are bell-shaped with long tails, so at 4 bits a min-max grid
spends most of its 15 levels on a region that holds almost no values. Every serious
quantization pipeline therefore searches for a smaller range on calibration data:
GPTQ's quantizer has an option that shrinks the range in steps of 1 percent and keeps
the best, AWQ follows its scaling search with exactly this kind of grid search over
clipping fractions, and static activation quantizers pick ranges with MSE, percentile,
or entropy calibrators. The search is simple, and each of its three bugs is easy to
ship: an error that only counts clipping never clips, an error without the clamp
always clips as hard as it can, and a range computed per calibration sample is one
that the deployed model, which has a single stored scale, cannot use.
