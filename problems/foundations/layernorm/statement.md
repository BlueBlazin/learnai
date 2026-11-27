Implement LayerNorm, the normalization layer used in GPT-2, BERT, and modern embedding and reranking encoders. For each vector, subtract its mean, divide by its standard deviation, and apply a learned gain and bias. LayerNorm differs from RMSNorm in three details (mean subtraction, population variance, and a bias term), and getting any one of them wrong still produces plausible-looking activations.

## Specification

```python
def layer_norm(x, weight, bias, eps=1e-5): ...
```

Given an input vector $x \in \mathbb{R}^d$, a learned gain $g \in \mathbb{R}^d$ (`weight`), a learned bias $b \in \mathbb{R}^d$ (`bias`), and a positive stability constant $\varepsilon$ (`eps`), LayerNorm computes

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,
\qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2,
\qquad
\operatorname{layer\_norm}(x)_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}\; g_i + b_i .
$$

Follow these three conventions:

- Use the **biased** (population) variance, dividing the sum of squared deviations by $d$ rather than $d - 1$.
- Add `eps` to the **variance, inside the square root**.
- Multiply the normalized vector by `weight` first, and add `bias` last.

For a multi-dimensional tensor `x`, apply this transformation independently to every vector along the **last axis**. All leading axes are batch axes (such as batch and sequence), so each vector has its own mean and variance. Here `eps` is a positive Python float (default `1e-5`).

A vector whose entries are all equal has $\sigma^2 = 0$. Its normalized form is the zero vector, so the output is `bias`. This includes every vector when $d = 1$.

Write the arithmetic yourself. Do not call a built-in normalization layer or function (`torch.nn.LayerNorm`, `torch.nn.functional.layer_norm`, `jax.nn.standardize`, `mlx.nn.LayerNorm`, `mx.fast.layer_norm`, and similar). Reductions such as `mean` and `var` are allowed, but check which variance your framework computes by default.

## Shapes

| name | shape | dtype |
|---|---|---|
| `x` | `(..., d)`, one to four dimensions | float32 |
| `weight` | `(d,)` | float32 |
| `bias` | `(d,)` | float32 |
| result | same as `x` | float32 |

## Example

```python
weight = [1.0, 1.0, 2.0, 0.5]
bias = [0.0, 1.0, 0.0, -1.0]

layer_norm([1.0, 2.0, 3.0, 6.0], weight, bias)  # [-1.069043, 0.465478, 0.0, -0.198217]
layer_norm([101.0, 102.0, 103.0, 106.0], weight, bias)  # the same: a shift cancels
layer_norm([5.0, 5.0, 5.0, 5.0], weight, bias)  # [0.0, 1.0, 0.0, -1.0], the bias
```

For `[1, 2, 3, 6]`, the mean is $3$ and the deviations are $[-2, -1, 0, 3]$. The biased variance is $(4 + 1 + 0 + 9) / 4 = 3.5$, which gives a standard deviation of $1.870829$ and a normalized vector of $[-1.069043, -0.534522, 0, 1.603565]$. Applying `weight` and `bias` produces the first output. Dividing by $d - 1 = 3$ instead would give a variance of $4.667$ and a first entry of $-0.925819$.

`eps` has little effect on that example, but it is decisive for small activations. For `[0.001, 0.002, 0.003, 0.006]`, the variance is $3.5 \times 10^{-6}$, which is smaller than $\varepsilon = 10^{-5}$. With unit gain and zero bias, the result is `[-0.544331, -0.272166, 0.0, 0.816497]`. Adding `eps` outside the square root instead would give `[-1.063361, -0.531681, 0.0, 1.595042]`, almost twice as large.

## Constraints

- Results are compared against the reference with relative tolerance `1e-4` and absolute tolerance `1e-5`. A NaN or an infinity is always wrong.
- Every entry of `x` is finite with $|x_i| \le 100$, so squaring does not overflow here (unlike RMSNorm). Entries of `weight` and `bias` satisfy $|g_i|, |b_i| \le 10$.
- $10^{-8} \le \varepsilon \le 10^{-2}$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32.

## Why this matters

Decoder-only LLMs of the Llama family moved to RMSNorm, but LayerNorm is still what you port when serving GPT-2, BERT-style embedding models, rerankers, Whisper, or a vision encoder in front of a multimodal model. These conventions are baked into the trained weights. An inference engine that divides by $d - 1$ scales every activation by $\sqrt{(d-1)/d}$, which is 0.9993 at $d = 768$. That error is too small to spot in a smoke test, yet it compounds at every layer. Placing `eps` outside the square root only surfaces on low-variance vectors, such as padding positions or the near-constant activations of a saturated layer.
