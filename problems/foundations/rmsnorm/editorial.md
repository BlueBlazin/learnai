## The idea

RMSNorm drops the mean subtraction of LayerNorm and keeps only the rescaling: divide by
the root mean square, multiply by a gain. The direct translation is

```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
```

and it is correct whenever $x^2$ is representable. In float32 that means $|x| <
1.8 \times 10^{19}$. Beyond that the square is `inf`, the reciprocal root is `0`, and the
output is a vector of zeros where the true answer has entries of order 1.

The fix uses the fact that the function is nearly scale-invariant. For any $s > 0$:

$$
\frac{x}{\sqrt{\operatorname{mean}(x^2) + \varepsilon}}
= \frac{x/s}{\sqrt{\operatorname{mean}\big((x/s)^2\big) + \varepsilon/s^2}} .
$$

Multiply the numerator and the denominator of the left side by $1/s$ and move it inside
the root to see it. Choosing $s = \max(1, \max_j |x_j|)$ per vector gives $u = x / s$
with every entry in $[-1, 1]$:

- **No overflow.** Squares of $u$ are at most 1.
- **No loss for large vectors.** When $s > 1$ the largest entry of $u$ has magnitude 1,
  so $\operatorname{mean}(u^2) \ge 1/d$. The term $\varepsilon / s^2$ may round to zero
  (or $s^2$ may overflow to `inf`, giving exactly zero); either way it is far below
  float32 resolution next to $1/d$.
- **Nothing changes for small vectors.** When $\max |x_j| \le 1$, $s = 1$ and this is
  the direct formula. Squares of tiny entries may underflow to zero, which is fine:
  they are being added to $\varepsilon$, which is at least $10^{-8}$.
- **Zeros stay zeros.** For an all-zero vector the result is $0 / \sqrt{\varepsilon} = 0$.

The floor of 1 on $s$ is what makes the last two points work. Dividing by the plain
maximum would need a special case for the zero vector.

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

- **Epsilon outside the square root**, `x / (rms + eps)`. This is the formula of the
  original RMSNorm paper's reference code, and it is not what Llama checkpoints use.
  The two agree to about $\varepsilon / \text{rms}$ for ordinary activations and differ
  by tens of percent once the root mean square approaches $\sqrt{\varepsilon} \approx
  0.003$. The same trap exists between other normalization layers, so always check
  which one a checkpoint was trained with.
- **No epsilon at all.** An all-zero vector (padding, a dead residual stream at
  initialization) becomes `0 / 0 = NaN`, and one NaN poisons everything downstream.
- **Squaring before scaling.** Correct until the activations are large, then silently
  all zeros.
- **Subtracting the mean first.** That is LayerNorm. The outputs look plausible, so
  this survives casual inspection.
- **Sum instead of mean.** Dividing by the L2 norm instead of the root mean square
  shrinks the output by $\sqrt{d}$.
- **Reducing over the wrong axes.** Normalizing over sequence and feature axes together
  passes every test with a `(batch, d)` input and couples the tokens of a sequence in
  the real model.
- **Applying the gain before normalizing.** With a gain of ones (the initialization)
  it is identical, so the bug appears only with trained weights.

## Complexity

One pass for the maximum and one for the mean square along the last axis: $O(d)$ time
per vector and $O(d)$ extra memory for the scaled copy.

## In the wild

Meta's Llama reference code computes exactly the direct formula,
`x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`, with `eps = 1e-5`, and it
guards against overflow in a different way: it casts to float32 first (`x.float()`) and
casts the result back, because models are served in float16 or bfloat16. Float16 tops
out at 65504, so a square overflows as soon as an activation exceeds 255, well within
what residual streams reach. Computing in a wider type is the standard answer when one
is available. The rescaling trick in this problem is the answer when it is not: it is
the same idea that makes `hypot` and the BLAS 2-norm safe, and it keeps the whole
computation in the working precision.

The reciprocal square root is also worth noticing. Engines prefer `x * rsqrt(...)` to
`x / sqrt(...)` because one `rsqrt` and a multiply is cheaper than a square root and a
division per element, and RMSNorm runs twice per layer for every token.
