## The idea

Both functions are invariant to shifting the logits along the axis by a constant $c$:

$$
\frac{e^{x_i - c}}{\sum_j e^{x_j - c}}
= \frac{e^{-c}\, e^{x_i}}{e^{-c} \sum_j e^{x_j}}
= \frac{e^{x_i}}{\sum_j e^{x_j}} .
$$

The textbook formula is the special case $c = 0$, and it is the worst possible choice
when the logits are far from zero. Choosing $c = m = \max_j x_j$ fixes both failure
modes at once:

- **No overflow.** Every shifted logit is $\le 0$, so every exponential is in $[0, 1]$.
- **No `0 / 0`.** The maximal entry becomes $e^0 = 1$, so the denominator is in $[1, n]$.

Small terms may still underflow to zero, and that is fine. A term that underflows is
smaller than about $10^{-45}$ relative to the largest one, far below float32 resolution,
so dropping it does not change the rounded result.

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

`log(softmax(x))` is mathematically identical and numerically wrong. A probability
below about $10^{-45}$ is not representable in float32 and rounds to `0`, whose log is
`-inf`, while the log-probability itself (say `-200`) is a perfectly ordinary float.
The fix is to never leave log space:

$$
\operatorname{log\_softmax}(x)_i = (x_i - m) - \log \sum_j e^{x_j - m}.
$$

The second term is the log of a number in $[1, n]$, hence in $[0, \log n]$. Nothing can
overflow, and the only way to get `-inf` out is to put `-inf` in.

That last case is intended. A masked logit of `-inf` gives a shifted logit of `-inf`,
an exponential of exactly `0`, a probability of `0`, and a log-probability of `-inf`.
No special handling is needed, as long as the slice has one finite entry to serve as
the maximum.

## Common mistakes

- **Reducing without keeping the dimension.** `x.max(dim=axis)` drops the axis, and
  the subtraction then either fails to broadcast or, for square inputs, silently
  broadcasts along the wrong axis.
- **Subtracting the global maximum.** The shift must be per slice. With a global
  maximum, a row whose logits all sit far below it underflows to `0 / 0`.
- **Adding an epsilon** (`log(p + 1e-8)`). This hides the `-inf` but returns a wrong
  number: every log-probability below about `-18.4` is clamped to it.
- **Clamping the logits** before exponentiating. It avoids the overflow by changing
  the answer.

## Complexity

Two passes over the data along the axis (a max and a sum), $O(n)$ time, and $O(n)$
extra memory for the shifted copy.

## In the wild

This is exactly the "log-sum-exp trick": $\log \sum_j e^{x_j} = m + \log \sum_j
e^{x_j - m}$. You will use it again for sampling with temperature, for nucleus (top-p)
filtering, for beam search scores, and for cross-entropy, all of which are best computed
from `log_softmax` and not from probabilities.
