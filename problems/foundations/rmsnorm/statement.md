Implement RMSNorm, the normalization layer of Llama-family models: divide each vector by
its root mean square, then multiply by a learned gain. The formula is one line; the work
is in getting the epsilon in the right place and in returning the right answer for
activations so large that their squares do not fit in float32.

![This problem is RMSNorm, which a decoder-only transformer applies before attention, before the MLP, and before the output layer.](assets/map.svg)

## Specification

```python
def rms_norm(x, weight, eps=1e-5): ...
```

For a vector $x \in \mathbb{R}^d$ and a gain $g \in \mathbb{R}^d$:

$$
\operatorname{rms\_norm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \; g_i .
$$

The epsilon is added to the **mean square, inside the square root**. This is the
convention of the Llama reference implementation, and it is the one checkpoints are
trained with.

For a tensor, apply the function independently to every vector along the **last axis**.
All other axes are batch axes (batch, sequence, and so on): each vector gets its own
root mean square.

- `x`: a float32 tensor of shape `(..., d)` with one to four dimensions.
- `weight`: the gain $g$, a float32 tensor of shape `(d,)`. It multiplies the normalized
  vector elementwise, after the normalization.
- `eps`: a positive Python float, $\varepsilon$ above.
- Returns a float32 tensor with the same shape as `x`.

The result must be correct for every finite float32 input in the range given under
Constraints, including vectors whose squares overflow float32 (which happens once
$|x_i|$ exceeds about $1.8 \times 10^{19}$) and all-zero vectors, which come out as zeros.

Write the arithmetic yourself: do not call a built-in normalization layer or function
(`torch.nn.RMSNorm`, `torch.nn.functional.rms_norm`, `mlx.nn.RMSNorm`,
`mx.fast.rms_norm`, and the like).

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, one to four dimensions | float32 |
| `weight` | `(d,)` | float32 |
| result | same as `x` | float32 |

## Example

```python
weight = [1.0, 0.5]

rms_norm([3.0, 4.0], weight)  # [0.848528, 0.565685]
rms_norm([3e-3, 4e-3], weight)  # [0.632456, 0.421637]
rms_norm([3e20, 4e20], weight)  # [0.848528, 0.565685]
rms_norm([0.0, 0.0], weight)  # [0.0, 0.0]
```

For `[3, 4]` the mean square is $(9 + 16) / 2 = 12.5$ and the root mean square is
$3.5355$, so the normalized vector is $[0.8485, 1.1314]$ and the gain halves the second
entry. Epsilon changes the sixth decimal place at most.

For `[3e-3, 4e-3]` the mean square is $1.25 \times 10^{-5}$, about the size of
$\varepsilon = 10^{-5}$. The denominator is $\sqrt{2.25 \times 10^{-5}} = 4.743 \times
10^{-3}$, and the output is visibly smaller than in the first line. Adding epsilon
outside the square root instead would give `[0.846135, 0.564090]`.

For `[3e20, 4e20]` the exact answer is the same as for `[3, 4]`, but $x_i^2 \approx
10^{41}$ is larger than the largest float32 ($3.4 \times 10^{38}$). Squaring first gives
`inf`, and `x / inf` returns `[0.0, 0.0]`.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-6`. A NaN or an infinity is always wrong.
- Every entry of `x` is finite with $|x_i| \le 10^{30}$. Entries of `weight` satisfy
  $|g_i| \le 10$.
- $10^{-8} \le \varepsilon \le 10^{-2}$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32. (MLX has no float64 on the GPU, so "compute it in double precision"
  is not a portable answer.)

## Why this matters

A Llama-style decoder applies RMSNorm twice per layer and once more before the output
head, so every token passes through it dozens of times. The epsilon placement is part of
the trained model: an engine that puts it outside the square root reproduces the
reference activations to several digits on ordinary inputs and then drifts on the
low-magnitude ones, which is the kind of mismatch that survives a smoke test. Overflow
of the squares is not hypothetical either. In half precision, the format most engines
serve in, a square overflows once an activation exceeds 255, the normalized vector
collapses to zeros, and the model emits garbage from that layer on. Engines avoid it by
computing the norm at higher precision or, as here, by arranging the arithmetic so that
the squares stay in range.
