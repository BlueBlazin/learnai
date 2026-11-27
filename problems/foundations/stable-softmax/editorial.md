## The idea

For a 1-D slice $x \in \mathbb{R}^n$, both `softmax` and `log_softmax` are invariant to
subtracting any constant $c$ from every logit along the axis:

$$
\frac{e^{x_i - c}}{\sum_{j=1}^{n} e^{x_j - c}}
= \frac{e^{-c}\, e^{x_i}}{e^{-c} \sum_{j=1}^{n} e^{x_j}}
= \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} .
$$

The textbook formula sets $c = 0$, which overflows on large positive logits and
underflows to `0 / 0` on large negative logits. Choosing $c = m = \max_j x_j$ fixes both
problems at once:

- **No overflow.** Every shifted logit $x_i - m$ is at most `0`, so every exponential
  lies in $[0, 1]$.
- **No `0 / 0`.** The largest entry becomes $e^0 = 1$, so the denominator lies in
  $[1, n]$.

Very small terms may still underflow to `0`, and that is harmless. A term that underflows
is smaller than about $10^{-45}$ relative to the largest entry, which is far below
float32 precision and does not affect the rounded result.

## Reference solution (PyTorch)

```python
def softmax(x, axis=-1):
    shifted = x - torch.amax(x, dim=axis, keepdim=True)
    e = torch.exp(shifted)
    return e / e.sum(dim=axis, keepdim=True)


def log_softmax(x, axis=-1):
    shifted = x - torch.amax(x, dim=axis, keepdim=True)
    return shifted - torch.log(torch.exp(shifted).sum(dim=axis, keepdim=True))
```

The JAX and MLX versions are the same three lines with `keepdims=True`.

## Why `log_softmax` is its own function

Computing `log(softmax(x))` is mathematically equivalent to `log_softmax(x)`, but
numerically wrong. A probability below about $10^{-45}$ underflows to `0` in float32, and
taking its logarithm gives `-inf` even though the true log-probability (such as `-200`)
is an ordinary float32 value.

The fix is to stay in log space:

$$
\operatorname{log\_softmax}(x)_i = (x_i - m) - \log \sum_{j=1}^{n} e^{x_j - m}.
$$

The sum lies in $[1, n]$, so its logarithm lies in $[0, \log n]$. Nothing can overflow,
and `-inf` appears in the output only when `-inf` appears in the input.

That `-inf` behavior is exact for masked tokens. A logit of `-inf` gives a shifted logit
of `-inf`, an exponential of `0`, a probability of `0`, and a log-probability of `-inf`.
No special-case code is needed as long as each slice has at least one finite entry to
serve as the maximum.

## Common mistakes

- **Reducing without keeping the dimension.** Calling `x.max(dim=axis)` drops the axis,
  so the subtraction either fails to broadcast or silently broadcasts along the wrong
  axis on square inputs.
- **Subtracting the global maximum.** The shift must be computed per slice. With a global
  maximum, a slice whose logits all sit far below that maximum underflows to `0 / 0`.
- **Adding an epsilon (`log(p + 1e-8)`).** This hides `-inf`, but clamps every
  log-probability below about `-18.4` to a wrong value.
- **Clamping the logits before exponentiating.** Clamping avoids overflow by changing the
  mathematical answer.

## Complexity

Two passes over the data along `axis` (a max and a sum), taking $O(n)$ time per slice and
$O(n)$ extra memory for the shifted tensor.

## In the wild

This identity, $\log \sum_j e^{x_j} = m + \log \sum_j e^{x_j - m}$, is known as the
log-sum-exp trick. It shows up throughout LLM inference and training, including
temperature scaling, nucleus (top-p) filtering, beam search scoring, and cross-entropy
loss, all of which are safer to compute from `log_softmax` than from raw probabilities.
