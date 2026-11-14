## The idea

Both activations have the form "the input times a smooth gate between 0 and 1". They
differ in the gate.

**GELU** gates with the standard normal CDF, $\Phi(x) = P(Z \le x)$. The error function
is the same integral with a different scaling, $\operatorname{erf}(z) =
\frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2}\,dt$; substituting $t = u/\sqrt{2}$ gives

$$
\Phi(x) = \frac{1}{2}\left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right) .
$$

The $\sqrt{2}$ is the standard deviation hiding in the change of variables: `erf`
integrates $e^{-t^2}$, a Gaussian with variance $1/2$.

**The tanh form** replaces $\Phi$ with $\frac{1}{2}(1 + \tanh(\sqrt{2/\pi}\,(x +
0.044715\,x^3)))$. It was attractive in 2016 because `erf` was slow or missing on some
accelerators. Its maximum error is $4.7 \times 10^{-4}$, near $|x| = 2.7$. That is a
good approximation and a different function: a model trained with one and served with
the other is a slightly different model.

**SiLU** gates with the logistic sigmoid. Of the two ways to write the sigmoid, only
one survives float32:

| | $x = 100$ | $x = -100$ |
|---|---|---|
| $1 / (1 + e^{-x})$ | $1 / (1 + 0) = 1$ | $1 / (1 + \infty) = 0$ |
| $e^{x} / (1 + e^{x})$ | $\infty / \infty$ = NaN | $0 / (1 + 0) = 0$ |

Overflow to infinity is harmless in a denominator and fatal on both sides of a
division. Folding the multiplication in gives the one-line form $x / (1 + e^{-x})$.

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

The JAX version uses `jax.scipy.special.erf` and the MLX version `mx.erf`; nothing else
changes.

The exact form computes $1 + \operatorname{erf}(\cdot)$, which cancels for negative
$x$: below about $x = -5.5$ the `erf` rounds to exactly $-1$ and the result is $-0.0$
where the true value is of order $-10^{-7}$. The absolute error is below $10^{-6}$,
which is why the comparison has an absolute tolerance. If the tail matters, use the
complementary error function, $\Phi(x) = \frac{1}{2}\operatorname{erfc}(-x/\sqrt{2})$,
which has no cancellation (PyTorch and JAX have `erfc`; MLX does not).

## Common mistakes

- **Returning the tanh form for `approximate="none"`**, or the reverse. Typically the
  argument is ignored because the first formula found online was the tanh one. The
  outputs agree to three decimal places, so a loose test passes.
- **A wrong constant.** `sqrt(2) / pi` or `2 / sqrt(pi)` in place of `sqrt(2 / pi)`;
  `0.44715` in place of `0.044715`. The result is still a smooth gate with the right
  limits, so plots look right.
- **`erf(x)` without the $\sqrt{2}$.** This is $x$ times the CDF of a normal with
  variance $1/2$: a sharper gate, and the right values at 0 and at both extremes.
- **The other approximation.** The paper also offers $x\,\sigma(1.702\,x)$, with an
  error of up to 0.02. It has its own name in config files (`quick_gelu`) and is not
  what `approximate="tanh"` means.
- **The overflowing sigmoid.** `x * exp(x) / (1 + exp(x))` is NaN from $x = 88.73$ on.
  Activations that large are rare in float32, and common enough in the residual stream
  of a real model to matter.
- **Calling the built-in.** Correct numbers, and nothing learned about which function a
  config name refers to.

## Complexity

One pass over the data, $O(n)$ time and $O(n)$ memory, with one transcendental function
(`erf`, `tanh`, or `exp`) per element.

## In the wild

Hugging Face configuration files name the activation, and the names are a small
history lesson. `"gelu"` is the exact erf form. `"gelu_new"` is the tanh form written
out in Python, the default of GPT-2. `"gelu_pytorch_tanh"` is the same function through
PyTorch's `gelu(approximate="tanh")`, the value in Gemma configs; Gemma's first
release shipped with `"gelu"` although the weights were trained with the tanh form, and
the loader now corrects that value with a warning. `"quick_gelu"` is $x\,\sigma(1.702\,
x)$, used by the original CLIP. `"silu"` and `"swish"` are the same function; it is the
gate inside the SwiGLU feed-forward block of Llama-family models.

Engines implement each of these as a fused elementwise kernel, and in half precision
the overflow argument changes numbers but not shape: in float16, $e^{x}$ overflows at
$x = 11.1$ and $x^3$ at $x = 40.3$, so the stable spelling of the sigmoid is required
for ordinary activations, and the cubic in the tanh form needs care as well.
