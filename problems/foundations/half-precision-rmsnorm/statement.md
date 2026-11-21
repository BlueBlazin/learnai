Implement RMSNorm for a model served in half precision: float16 in, float16 out, and
float16 for everything in between. The formula has not changed since the `rmsnorm`
problem, but float16 tops out at 65504, so the square of any activation above 255 is
infinite, and its smallest normal number is $6.1 \times 10^{-5}$, so squares of small
activations and epsilon itself lose most of their bits.

![This problem is the same RMSNorm, run in float16 at each of its three places.](assets/map.svg)

## Specification

```python
def rms_norm_f16(x, weight, eps=1e-5): ...
```

For a vector $x \in \mathbb{R}^d$ and a gain $g \in \mathbb{R}^d$, as before:

$$
\operatorname{rms\_norm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \; g_i ,
$$

with epsilon added to the mean square inside the square root, applied independently to
every vector along the **last axis**.

- `x`: a float16 tensor of shape `(..., d)` with one to four dimensions. Its entries
  may be **any finite float16 values**: zero, subnormals such as $6 \times 10^{-8}$,
  and values up to the largest float16, $\pm 65504$.
- `weight`: the gain $g$, a float16 tensor of shape `(d,)`.
- `eps`: a positive Python float, $\varepsilon$ above.
- Returns a float16 tensor with the same shape as `x`.

**The value.** The expected result is the formula evaluated *exactly* on the float16
inputs (with `eps` as the Python float it is), then rounded to float16. Your result
must agree with it within the float16 tolerance under Constraints, for every input in
range. In particular a vector containing 60000 must not come out as zeros, and an
all-zero vector comes out as zeros.

**Float16 throughout.** Every floating-point tensor your function computes must be
float16. Do not convert `x`, `weight`, or an intermediate result to float32, float64, or
bfloat16, neither explicitly nor by combining it with a wider tensor (a float32
constant tensor promotes the result). What remains allowed:

- Python floats. `eps`, `math.sqrt(eps)`, and literals are scalars, not tensors, and a
  float16 tensor combined with a Python scalar stays float16.
- Your framework's reductions and elementwise functions applied to float16 tensors
  (`max`, `sum`, `mean`, `abs`, `sqrt`, ...), whatever they do internally.

The tests observe the dtypes of the intermediate results of your function, not only
the dtype of what it returns.

Write the arithmetic yourself: do not call a built-in normalization layer or function
(`torch.nn.RMSNorm`, `torch.nn.functional.rms_norm`, `mlx.nn.RMSNorm`,
`mx.fast.rms_norm`, and the like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, one to four dimensions | float16 |
| `weight` | `(d,)` | float16 |
| result | same as `x` | float16 |

## Example

Float16 in four numbers:

| | value |
|---|---|
| largest finite value | $65504$ |
| smallest normal value | $2^{-14} \approx 6.10 \times 10^{-5}$ |
| spacing of the subnormals below it | $2^{-24} \approx 5.96 \times 10^{-8}$ |
| precision | 11 bits, about 3 decimal digits |

```python
weight = [1.0, 0.5]

rms_norm_f16([3.0, 4.0], weight)  # [0.8485, 0.5657]
rms_norm_f16([30000.0, 40000.0], weight)  # [0.8485, 0.5657]
rms_norm_f16([3e-4, 4e-4], weight)  # [0.09426, 0.06286]
rms_norm_f16([0.0, 0.0], weight)  # [0.0, 0.0]
```

The first line is the example of `rmsnorm`: the root mean square is $3.5355$.

For `[30000, 40000]` the exact answer is the same, but $30000^2 = 9 \times 10^8$ is far
beyond 65504. Squaring first gives `inf`, and `x / inf` returns `[0.0, 0.0]`. The
float32 recipe of `rmsnorm` (divide by $\max(1, \max_j |x_j|)$ first) repairs this line.

It does not repair the small end. For `[3e-4, 4e-4]` the squares are $9 \times 10^{-8}$
and $1.6 \times 10^{-7}$, where float16 can only represent multiples of
$5.96 \times 10^{-8}$: they are stored as $1.19 \times 10^{-7}$ and $1.79 \times
10^{-7}$. Here $\varepsilon = 10^{-5}$ is a hundred times larger and hides the damage.
With a smaller epsilon it shows:

```python
rms_norm_f16([3e-4, -3e-4], [1.0, 1.0], eps=1e-6)  # [0.2873, -0.2873]
```

Computed naively in float16 this is `[0.2817, -0.2817]`, 2% off: the mean square
$9 \times 10^{-8}$ is stored as $1.19 \times 10^{-7}$, and $10^{-6}$ itself is stored
as $1.013 \times 10^{-6}$.

## Constraints

- Results are compared with the exact result rounded to float16, to a relative
  tolerance of `5e-3` and an absolute tolerance of `4e-6`. One float16 rounding is a
  relative error of up to $4.9 \times 10^{-4}$, so the relative tolerance allows for
  about ten roundings' worth of accumulated error; the reference solutions stay below
  60% of it on every input tried. The absolute tolerance covers outputs that are themselves
  near the bottom of the float16 range. A NaN or an infinity is always wrong.
- Every entry of `x` is a finite float16. Entries of `weight` satisfy
  $0.5 \le |g_i| \le 2$.
- $10^{-6} \le \varepsilon \le 10^{-2}$.
- $1 \le d \le 512$, and `x` has at most 100,000 elements. No Python loops over
  elements are needed.
- The result and every floating-point intermediate are float16, as specified above.

## Why this matters

Half precision is the default serving format on GPUs and on Apple silicon: it halves
memory and bandwidth. The residual stream of a real model does not stay small for it.
In the small model used elsewhere in this catalog (Granite 4.0 350M) the largest
hidden-state entry is about 9,400 after the third block, measured in float32; its
square is $8.8 \times 10^{7}$, more than a thousand times the float16 maximum. A naive
float16 RMSNorm turns such a vector into zeros, or into NaN once an `inf` meets another
`inf`, and every later layer inherits it. Engines avoid this either by computing the
norm in float32 and casting back, which costs a conversion and a wider temporary per
call, or by arranging the arithmetic so that it stays in range in the storage type.
This problem is the second way, which is the only one available when the wider type
is not (a fixed-point or half-only accelerator kernel), and the same rescaling idea
appears wherever a norm is computed near the edge of a number format.
