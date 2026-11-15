Implement LayerNorm, the normalization layer of GPT-2, BERT, and the encoders behind
today's embedding and reranking models: subtract each vector's mean, divide by its
standard deviation, then apply a learned gain and bias. It differs from RMSNorm in three
details (the mean subtraction, which variance is meant, and the bias), and a port that
gets one of them wrong still produces plausible activations.

## Specification

```python
def layer_norm(x, weight, bias, eps=1e-5): ...
```

For a vector $x \in \mathbb{R}^d$, a gain $g \in \mathbb{R}^d$, and a bias
$b \in \mathbb{R}^d$:

$$
\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,
\qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d} (x_j - \mu)^2,
\qquad
\operatorname{layer\_norm}(x)_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}\; g_i + b_i .
$$

- The variance is the **biased** (population) variance: the sum of squared deviations
  is divided by $d$, not by $d - 1$.
- The epsilon is added to the **variance, inside the square root**.
- The gain multiplies the normalized vector first, and the bias is added last.

For a tensor, apply the function independently to every vector along the **last axis**.
All other axes are batch axes (batch, sequence, and so on): each vector gets its own
mean and variance.

- `x`: a float32 tensor of shape `(..., d)` with one to four dimensions.
- `weight`: the gain $g$, a float32 tensor of shape `(d,)`.
- `bias`: the bias $b$, a float32 tensor of shape `(d,)`.
- `eps`: a positive Python float, $\varepsilon$ above.
- Returns a float32 tensor with the same shape as `x`.

A vector whose entries are all equal has $\sigma^2 = 0$. Its normalized form is the zero
vector, so the result is `bias`. This includes every vector when $d = 1$.

Write the arithmetic yourself: do not call a built-in normalization layer or function
(`torch.nn.LayerNorm`, `torch.nn.functional.layer_norm`, `jax.nn.standardize`,
`mlx.nn.LayerNorm`, `mx.fast.layer_norm`, and the like). Reductions such as `mean` and
`var` are allowed; check which variance yours computes.

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

For `[1, 2, 3, 6]` the mean is $3$, the deviations are $[-2, -1, 0, 3]$, and the biased
variance is $(4 + 1 + 0 + 9) / 4 = 3.5$, so the standard deviation is $1.870829$ and the
normalized vector is $[-1.069043, -0.534522, 0, 1.603565]$. The gain and the bias then
give the first line. Dividing by $d - 1 = 3$ instead would give a variance of $4.667$
and a first entry of $-0.925819$.

Epsilon is invisible in that example and decisive for small activations. For
`[0.001, 0.002, 0.003, 0.006]` the variance is $3.5 \times 10^{-6}$, smaller than
$\varepsilon = 10^{-5}$. With unit gain and zero bias the result is
`[-0.544331, -0.272166, 0.0, 0.816497]`; adding epsilon outside the square root
instead would give `[-1.063361, -0.531681, 0.0, 1.595042]`, almost twice as large.

## Constraints

- Results are compared with the reference to a relative tolerance of `1e-4` and an
  absolute tolerance of `1e-5`. A NaN or an infinity is always wrong.
- Every entry of `x` is finite with $|x_i| \le 100$; nothing overflows here, in contrast
  to RMSNorm. Entries of `weight` and `bias` satisfy $|g_i|, |b_i| \le 10$.
- $10^{-8} \le \varepsilon \le 10^{-2}$.
- `x` has at most 100,000 elements. No Python loops over elements are needed.
- Stay in float32.

## Why this matters

Decoder-only LLMs of the Llama family moved to RMSNorm, but LayerNorm is still what you
port when you serve GPT-2, BERT-style embedding models, rerankers, Whisper, or a vision
encoder in front of a multimodal model. The three details are part of the trained
weights. An engine that divides by $d - 1$ scales every activation by
$\sqrt{(d-1)/d}$, which is 0.9993 at $d = 768$: too small to see in a smoke test, and
applied again at every layer. Epsilon outside the root only shows on low-variance vectors, such as padding
positions and the near-constant activations of a saturated layer.
