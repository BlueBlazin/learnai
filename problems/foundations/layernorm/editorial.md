## The idea

LayerNorm (Ba, Kiros, and Hinton, 2016) standardizes each vector on its own: subtract
its mean, divide by its standard deviation, then let the model undo as much of that as
it wants with a learned gain and bias. Nothing depends on the other vectors in the
batch or the sequence, which is why it works unchanged at batch size 1 and at any
sequence length, and why inference needs no running statistics.

Three details separate it from RMSNorm:

| | RMSNorm | LayerNorm |
|---|---|---|
| centering | none | subtract the mean |
| scale statistic | $\sqrt{\frac1d \sum x_j^2 + \varepsilon}$ | $\sqrt{\frac1d \sum (x_j - \mu)^2 + \varepsilon}$ |
| learned parameters | gain | gain and bias |

Centering makes the output invariant to adding a constant to the input, and dividing
by the standard deviation makes it invariant to scaling (up to epsilon). RMSNorm keeps
only the second invariance, and saves one reduction and one subtraction per call.

The variance is the population variance, divided by $d$. The layer is not estimating
anything: the $d$ entries are the whole population, and this is what PyTorch computes
("the variance is calculated via the biased estimator, equivalent to
`torch.var(input, correction=0)`", `torch.nn.LayerNorm` documentation). Epsilon sits
inside the root, as in the same documentation's formula
$y = (x - \mathrm{E}[x]) / \sqrt{\mathrm{Var}[x] + \epsilon} \cdot \gamma + \beta$.

## Reference solution (PyTorch)

```python
def layer_norm(x, weight, bias, eps=1e-5):
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(var + eps) * weight + bias
```

The variance is computed from the already-centered vector, so the mean is taken once.
A constant vector gives `centered = 0` and `var = 0`, and the result is
`0 / sqrt(eps) * weight + bias = bias` without a special case: epsilon is what keeps
the division defined.

The JAX and MLX versions are the same three lines with `keepdims=True`.

## Common mistakes

- **The unbiased variance.** `torch.var` and `torch.std` divide by $d - 1$ by default
  (`correction=1`); `jnp.var` and `mx.var` divide by $d$ by default (`ddof=0`). Porting
  a NumPy or JAX line to PyTorch word for word therefore changes the answer by a factor
  $\sqrt{(d - 1)/d}$: 13% at $d = 4$, 0.07% at $d = 768$. The second number is why this
  survives a smoke test on a real model.
- **Epsilon outside the root**, `sqrt(var) + eps`. Identical for ordinary activations
  and visibly different once the variance is near $\varepsilon$: for a variance of
  $3.5 \times 10^{-6}$ the two denominators are $3.67 \times 10^{-3}$ and
  $1.88 \times 10^{-3}$.
- **No centering**, or centering only inside the variance. Both are RMSNorm-shaped
  bugs: the output then moves when a constant is added to the input. The second is the
  subtler one, since the variance itself is right.
- **Bias before gain.** `(normalized + bias) * weight` scales the bias. It is invisible
  with freshly initialized parameters (gain 1, bias 0) and wrong with trained ones.
- **Reducing over more than the last axis.** Normalizing over `(sequence, d)` makes
  every token depend on the others in its sequence, including padding.
- **Dropping epsilon.** A constant vector, which is what a padded or saturated position
  can look like, becomes `0 / 0`.

## Complexity

Two reductions and a handful of elementwise operations along the last axis: $O(d)$ time
per vector and $O(d)$ extra memory for the centered copy.

## In the wild

GPT-2, BERT and its descendants (the encoders behind most embedding and reranking
models), Whisper, and vision transformers all use LayerNorm, with $\varepsilon$ between
$10^{-12}$ (BERT) and $10^{-5}$ (GPT-2, the PyTorch default): read it from the model's
config instead of assuming it. Checkpoints call the parameters `weight` and `bias`
(PyTorch) or `gamma` and `beta` (the original TensorFlow BERT). Large inputs deserve a
note: with a mean far from zero, `mean(x^2) - mean(x)^2` cancels catastrophically in
float32, which is another reason to compute the variance from the centered vector as
above.
