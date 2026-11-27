Implement RMSNorm entirely in `float16`, keeping the inputs, output, and every
intermediate tensor in `float16`. The formula is the same as in the `rmsnorm` problem,
but `float16` overflows above 65504, so squaring any activation above 255 produces
infinity. Its smallest normal number is $6.1 \times 10^{-5}$, so squaring small
activations or adding a small stabilizing constant loses most of its precision.

![This problem is the same RMSNorm, run in float16 at each of its three places.](assets/map.svg)

## Specification

```python
def rms_norm_f16(x, weight, eps=1e-5): ...
```

Apply RMSNorm independently to every vector along the **last axis** of `x`. For a vector
$x \in \mathbb{R}^d$, a gain vector $g \in \mathbb{R}^d$ (`weight`), and a positive
Python float $\varepsilon$ (`eps`), RMSNorm adds $\varepsilon$ to the mean square inside
the square root, divides each entry by the result, and multiplies by the gain:

$$
\operatorname{rms\_norm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \; g_i .
$$

The entries of `x` may be any finite `float16` values, including zero, subnormals down to
$6 \times 10^{-8}$, and extreme values up to $\pm 65504$.

**Expected value.** The reference result is the exact formula evaluated on the `float16`
inputs (using `eps` as the Python float given) and rounded to `float16`. Your output must
agree with it within the tolerances in Constraints for every valid input. In particular,
a vector containing `60000` must not collapse to zeros, and an all-zero vector must
return zeros.

**Float16 throughout.** Every floating-point tensor your function creates must have dtype
`float16`. Do not convert `x`, `weight`, or any intermediate result to `float32`,
`float64`, or `bfloat16`, either explicitly or by combining it with a wider tensor (for
example, a default `float32` constant tensor promotes the result). The following are
allowed:

- Python floats such as `eps`, `math.sqrt(eps)`, and numeric literals. Combining a
  `float16` tensor with a Python scalar keeps the tensor in `float16`.
- Standard framework reductions and elementwise operations (`max`, `sum`, `mean`, `abs`,
  `sqrt`, and similar) called on `float16` tensors, regardless of what they do
  internally.

The test suite inspects the dtypes of intermediate tensors produced by your function, not
only the dtype of the returned tensor.

Write the arithmetic yourself. Do not call a built-in normalization layer or function
(`torch.nn.RMSNorm`, `torch.nn.functional.rms_norm`, `mlx.nn.RMSNorm`,
`mx.fast.rms_norm`, or equivalents).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, one to four dimensions | float16 |
| `weight` | `(d,)` | float16 |
| result | same as `x` | float16 |

## Example

Four numbers summarize the limits of `float16`:

| property | value |
|---|---|
| largest finite value | $65504$ |
| smallest normal value | $2^{-14} \approx 6.10 \times 10^{-5}$ |
| spacing of subnormals below it | $2^{-24} \approx 5.96 \times 10^{-8}$ |
| precision | 11 bits, about 3 decimal digits |

```python
weight = [1.0, 0.5]

rms_norm_f16([3.0, 4.0], weight)  # [0.8485, 0.5657]
rms_norm_f16([30000.0, 40000.0], weight)  # [0.8485, 0.5657]
rms_norm_f16([3e-4, 4e-4], weight)  # [0.09426, 0.06286]
rms_norm_f16([0.0, 0.0], weight)  # [0.0, 0.0]
```

The first call is the example from `rmsnorm`, where the root mean square is $3.5355$.

For `[30000.0, 40000.0]`, the exact answer is the same, but $30000^2 = 9 \times 10^8$
exceeds 65504. Squaring first produces `inf`, and `x / inf` returns `[0.0, 0.0]`. The
`float32` safeguard from `rmsnorm` (dividing by $\max(1, \max_j |x_j|)$ before squaring)
fixes this large-input case.

It does not fix the small end. For `[3e-4, 4e-4]`, the squares are $9 \times 10^{-8}$ and
$1.6 \times 10^{-7}$. In that range, `float16` can only represent multiples of
$5.96 \times 10^{-8}$, so the squares round to $1.19 \times 10^{-7}$ and
$1.79 \times 10^{-7}$. With the default $\varepsilon = 10^{-5}$, epsilon is a hundred
times larger than the squares and hides the rounding error. A smaller epsilon exposes it:

```python
rms_norm_f16([3e-4, -3e-4], [1.0, 1.0], eps=1e-6)  # [0.2873, -0.2873]
```

Evaluated naively in `float16`, this call returns `[0.2817, -0.2817]`, which is 2% off.
The mean square $9 \times 10^{-8}$ rounds to $1.19 \times 10^{-7}$, and $10^{-6}$ itself
rounds to $1.013 \times 10^{-6}$.

## Constraints

- Outputs are compared against the exact result rounded to `float16` with a relative
  tolerance of `5e-3` and an absolute tolerance of `4e-6`. A single `float16` rounding
  introduces up to $4.9 \times 10^{-4}$ relative error, so the relative tolerance allows
  about ten roundings of accumulated error. The reference solutions stay below 60% of
  this budget on every tested input. The absolute tolerance covers outputs near the
  bottom of the `float16` range. Any NaN or infinity fails the test.
- Every entry of `x` is a finite `float16`. Entries of `weight` satisfy
  $0.5 \le |g_i| \le 2$.
- $10^{-6} \le \varepsilon \le 10^{-2}$.
- $1 \le d \le 512$, and `x` has at most 100,000 elements. No Python loops over elements
  are needed.
- The returned tensor and every floating-point intermediate tensor must be `float16`.

## Why this matters

Half precision is the default serving format on GPUs and Apple silicon because it halves
memory use and bandwidth. Real model activations often exceed its squaring limit. In
Granite 4.0 350M (used elsewhere in this catalog), the largest hidden-state entry reaches
about 9,400 after the third block when measured in `float32`. Its square is
$8.8 \times 10^{7}$, over a thousand times larger than the `float16` maximum. A naive
`float16` RMSNorm turns that vector into zeros, or into NaN if two infinities interact,
and every downstream layer inherits the corruption.

Inference engines avoid this either by upcasting to `float32` for the norm and casting
back, or by rescaling the arithmetic to stay in range within the storage type. Upcasting
adds conversions and wider temporary buffers. Rescaling works even when no wider type is
available, such as in half-only or fixed-point accelerator kernels, and the same pattern
appears wherever a norm is computed near the limits of a floating-point format.
