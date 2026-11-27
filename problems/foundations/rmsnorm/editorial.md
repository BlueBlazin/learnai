## The idea

RMSNorm drops the mean subtraction of LayerNorm and keeps only the rescaling. It divides
each vector $x \in \mathbb{R}^d$ by its root mean square and multiplies by the gain. The
direct translation is:

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

This works whenever $x^2$ fits in float32, which holds for $|x| < 1.8 \times 10^{19}$.
Above that threshold, $x^2$ overflows to `inf`, the reciprocal square root becomes `0`,
and the output collapses to zeros even though the true result has entries of order 1.

The fix uses the near scale-invariance of RMSNorm. For any positive scale $s > 0$,
dividing the numerator and denominator by $s$ and moving $1/s$ inside the square root
gives:

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{x/s}{\sqrt{\operatorname{mean}\big((x/s)^2\big) + \varepsilon/s^2}} .
$$

Choosing $s = \max(1, \max_j |x_j|)$ per vector produces $u = x / s$ with every entry in
$[-1, 1]$:

- **No overflow.** Every entry of $u^2$ is at most 1.
- **No precision loss for large vectors.** When $s > 1$, the largest entry of $u$ has
  magnitude 1, so $\operatorname{mean}(u^2) \ge 1/d$. The term $\varepsilon / s^2$ may
  underflow to `0` (or $s^2$ may overflow to `inf`, yielding `0`), which is far below
  float32 precision relative to $1/d$.
- **Unchanged for small vectors.** When $\max_j |x_j| \le 1$, we have $s = 1$, which
  reduces to the direct formula. Tiny squares may underflow to `0`, which is harmless
  because they are added to $\varepsilon \ge 10^{-8}$.
- **Zeros stay zeros.** For an all-zero vector, $s = 1$ and the result is
  $0 / \sqrt{\varepsilon} = 0$.

Clamping $s$ at a floor of 1 is what makes the last two properties work. Dividing by
$\max_j |x_j|$ directly would require a special case for the zero vector.

## Reference solution (PyTorch)

```python
def rms_norm(x, weight, eps=1e-5):
    s = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1.0)
    u = x / s
    mean_sq = u.square().mean(dim=-1, keepdim=True)
    return u / torch.sqrt(mean_sq + eps / (s * s)) * weight
```

The JAX and MLX versions are line-for-line translations with `keepdims=True`.

## Common mistakes

- **Epsilon outside the square root (`x / (rms + eps)`).** The original RMSNorm paper's
  reference code used this form, but Llama checkpoints do not. The two agree to about
  $\varepsilon / \text{rms}$ on ordinary activations and diverge by tens of percent once
  the root mean square approaches $\sqrt{\varepsilon} \approx 0.003$. Always check which
  convention a checkpoint was trained with.
- **No epsilon at all.** An all-zero vector (such as padding or a dead residual stream at
  initialization) produces `0 / 0 = NaN`, which corrupts everything downstream.
- **Squaring before scaling.** This passes ordinary inputs and then silently returns all
  zeros on large activations.
- **Subtracting the mean first.** That computes LayerNorm instead of RMSNorm. The outputs
  look plausible, so this bug can survive casual inspection.
- **Summing instead of taking the mean.** Dividing by the L2 norm rather than the root
  mean square shrinks the output by $\sqrt{d}$.
- **Reducing over the wrong axes.** Normalizing across sequence and feature axes together
  passes `(batch, d)` tests, but mixes tokens together in a 3-D sequence tensor.
- **Applying the gain before normalizing.** When `weight` is initialized to all ones, the
  bug is invisible. It only shows up with trained weights.

## Complexity

One pass along the last axis to find the maximum and one pass to compute the mean square,
taking $O(d)$ time per vector and $O(d)$ extra memory for the scaled copy.

## In the wild

Meta's Llama reference code runs the direct formula,
`x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`, with `eps = 1e-5`. Because
models are served in float16 or bfloat16, it prevents overflow by casting `x` to float32
(`x.float()`) and casting the output back. Float16 tops out at 65,504, so a square
overflows as soon as an activation exceeds 255, which residual streams routinely reach.
Upcasting to a wider dtype is the standard fix when one is available. When it is not, the
rescaling trick in this problem keeps the entire calculation in working precision, using
the same idea behind safe `hypot` and the BLAS 2-norm.

The reciprocal square root is also worth noting. Engines prefer `x * rsqrt(...)` over
`x / sqrt(...)` because one `rsqrt` and a multiply per element is cheaper than a square
root and a division, and RMSNorm runs twice per layer for every token.
