Implement `rms_norm`, the normalization layer used in Llama-family models. It divides
each vector by its root mean square and multiplies the result by a learned gain, while
staying accurate even when squared activations would overflow float32.

![This problem is RMSNorm, which a decoder-only transformer applies before attention, before the MLP, and before the output layer.](assets/map.svg)

## Specification

```python
def rms_norm(x, weight, eps=1e-5): ...
```

For a 1-D vector $x \in \mathbb{R}^d$, a learned gain $g \in \mathbb{R}^d$ (passed as
`weight`), and a positive Python float $\varepsilon$ (passed as `eps`), RMSNorm computes:

$$
\operatorname{rms\_norm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \; g_i .
$$

Add $\varepsilon$ to the **mean square, inside the square root**. This matches the Llama
reference implementation and the convention checkpoints are trained with.

For a multi-dimensional tensor `x`, apply this normalization independently to every
vector along the **last axis**. All leading axes are batch axes (such as batch and
sequence), so each vector gets its own root mean square. The gain `weight` broadcasts
across the leading axes and multiplies the normalized vector elementwise.

The output must be accurate for every finite float32 input within the constraints below.
This includes large vectors whose squares overflow float32 (which happens once $|x_i|$
exceeds about $1.8 \times 10^{19}$) and all-zero vectors, which must return zeros.

Write the arithmetic yourself. Do not call a built-in normalization layer or function
(`torch.nn.RMSNorm`, `torch.nn.functional.rms_norm`, `mlx.nn.RMSNorm`,
`mx.fast.rms_norm`, or similar).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, 1 to 4 dimensions | float32 |
| `weight` | `(d,)` | float32 |
| result | `(..., d)` (same as `x`) | float32 |

## Example

```python
weight = [1.0, 0.5]

rms_norm([3.0, 4.0], weight)  # [0.848528, 0.565685]
rms_norm([3e-3, 4e-3], weight)  # [0.632456, 0.421637]
rms_norm([3e20, 4e20], weight)  # [0.848528, 0.565685]
rms_norm([0.0, 0.0], weight)  # [0.0, 0.0]
```

For `[3.0, 4.0]`, the mean square is $(9 + 16) / 2 = 12.5$ and the root mean square is
$3.5355$. The normalized vector is `[0.8485, 1.1314]`, and the gain halves the second
entry. Here $\varepsilon$ affects at most the sixth decimal place.

For `[3e-3, 4e-3]`, the mean square is $1.25 \times 10^{-5}$, which is comparable to
$\varepsilon = 10^{-5}$. The denominator is $\sqrt{2.25 \times 10^{-5}} = 4.743 \times
10^{-3}$, so the output is noticeably smaller than in the first call. Adding
$\varepsilon$ outside the square root instead would give `[0.846135, 0.564090]`.

For `[3e20, 4e20]`, the exact answer matches `[3.0, 4.0]`, but $x_i^2 \approx 10^{41}$
exceeds the largest float32 value ($3.4 \times 10^{38}$). Squaring directly overflows to
`inf`, and `x / inf` returns `[0.0, 0.0]`.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`. A `NaN` or infinity is always wrong.
- Every entry of `x` is finite with $|x_i| \le 10^{30}$. Entries of `weight` satisfy
  $|g_i| \le 10$.
- $10^{-8} \le \varepsilon \le 10^{-2}$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32. MLX does not support float64 on the GPU, so upcasting to float64 is
  not a portable solution.

## Why this matters

A Llama-style decoder applies RMSNorm twice per layer and once more before the output
head, so every token passes through it dozens of times. Because the $\varepsilon$
placement is baked into the trained weights, putting it outside the square root matches
ordinary activations to several digits and then drifts on low-magnitude ones. That kind
of mismatch easily slips past a smoke test.

Square overflow is also a real serving failure. Most inference engines run in float16,
where a square overflows as soon as an activation exceeds 255. When that happens, the
normalized vector collapses to zeros and the model emits garbage from that layer onward.
Engines avoid this either by computing the norm in a wider dtype or, as in this problem,
by arranging the arithmetic so the squares stay in range.
