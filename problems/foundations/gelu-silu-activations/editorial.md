## The idea

Both activations multiply their input by a smooth gate between 0 and 1. They differ in which gate they use.

**GELU** gates with the standard normal CDF, $\Phi(x) = P(Z \le x)$ for a standard normal random variable $Z$. The error function is the same integral with a different scaling:

$$
\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2}\,dt .
$$

Substituting $t = u/\sqrt{2}$ converts that integral into the standard normal CDF:

$$
\Phi(x) = \frac{1}{2}\left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right) .
$$

The $\sqrt{2}$ factor comes from that change of variables, because `erf` integrates $e^{-t^2}$, which corresponds to a Gaussian with variance $1/2$ rather than $1$.

**The `tanh` approximation** replaces $\Phi(x)$ with $\frac{1}{2}(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3)))$. It became popular in 2016 because `erf` was slow or unavailable on some accelerators. Its maximum difference from exact GELU is $4.7 \times 10^{-4}$ near $|x| = 2.7$. That makes it a close approximation, but still a distinct function. A model trained with one form and served with the other produces slightly different activations.

**SiLU** gates with the logistic sigmoid. Only one of the two standard ways to write the sigmoid survives float32 for large inputs:

| formula | $x = 100$ | $x = -100$ |
|---|---|---|
| $1 / (1 + e^{-x})$ | $1 / (1 + 0) = 1$ | $1 / (1 + \infty) = 0$ |
| $e^{x} / (1 + e^{x})$ | $\infty / \infty$ = NaN | $0 / (1 + 0) = 0$ |

Overflow to infinity is harmless when it appears only in the denominator, but fatal when it appears in both the numerator and denominator. Folding the multiplication by $x$ into the fraction gives the single-line expression $x / (1 + e^{-x})$.

## Reference solution (PyTorch)

```python
def gelu(x, approximate="none"):
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + torch.tanh(inner))
    if approximate != "none":
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def silu(x):
    return x / (1.0 + torch.exp(-x))
```

The JAX version replaces `torch.erf` with `jax.scipy.special.erf`, and the MLX version uses `mx.erf`. Everything else stays the same.

The exact GELU formula computes $1 + \operatorname{erf}(\cdot)$, which suffers from catastrophic cancellation for negative $x$. Below roughly $x = -5.5$, `erf` rounds to `-1.0` in float32 and the formula returns `-0.0`, even though the true value is on the order of $-10^{-7}$. The absolute error stays below $10^{-6}$, which is why the test harness includes an absolute tolerance. When tail accuracy matters, you can use the complementary error function, $\Phi(x) = \frac{1}{2}\operatorname{erfc}(-x/\sqrt{2})$, which avoids cancellation. PyTorch and JAX provide `erfc`, while MLX does not.

## Common mistakes

- **Ignoring the `approximate` flag.** Returning the `tanh` form for `approximate="none"` (or the reverse) is common when someone copies the first GELU formula they find online. Because the two outputs agree to three decimal places, a loose check will miss the bug.
- **Getting a constant wrong.** Writing `sqrt(2) / pi` or `2 / sqrt(pi)` instead of `sqrt(2 / pi)`, or `0.44715` instead of `0.044715`, still yields a smooth gate with the right limits, so plots look normal even though the values are wrong.
- **Omitting the $\sqrt{2}$ inside `erf`.** Calling `erf(x)` computes $x$ times the CDF of a normal distribution with variance $1/2$. The resulting gate is too sharp, even though it matches at $x = 0$ and at both extremes.
- **Using `quick_gelu` for `approximate="tanh"`.** The original paper also mentions $x\,\sigma(1.702\,x)$, which has an error of up to $0.02$. Config files name that variant `quick_gelu`, not `approximate="tanh"`.
- **Using the overflowing sigmoid.** Computing `x * exp(x) / (1 + exp(x))` returns NaN starting at $x = 88.73$. Activations that large are uncommon in float32, yet frequent enough in the residual stream of real models to break inference.
- **Calling a built-in activation.** Built-in calls produce the right numbers, but bypass the implementation rules and hide which mathematical function a config name actually selects.

## Complexity

For an input with $n$ elements, one pass takes $O(n)$ time and $O(n)$ memory, with one transcendental call (`erf`, `tanh`, or `exp`) per element.

## In the wild

Hugging Face configuration files record the activation name, and those names reflect how the variants evolved. `"gelu"` selects the exact `erf` form. `"gelu_new"` is the `tanh` form written out in Python, which was the default for GPT-2. `"gelu_pytorch_tanh"` is the same `tanh` function routed through PyTorch's `gelu(approximate="tanh")`, as used in Gemma configs. Gemma's initial release accidentally specified `"gelu"` even though the weights were trained with the `tanh` form, so loaders now override that setting with a warning. `"quick_gelu"` is $x\,\sigma(1.702\,x)$, used in the original CLIP models. `"silu"` and `"swish"` refer to the same function, which serves as the gate inside the SwiGLU feed-forward block of Llama-family models.

Production inference engines implement these activations as fused elementwise kernels. In half precision (float16), overflow thresholds drop sharply. Because $e^{x}$ overflows at $x = 11.1$ and $x^3$ overflows at $x = 40.3$, the stable form of the sigmoid is necessary even for ordinary activations, and the cubic term in the `tanh` GELU requires extra care as well.
