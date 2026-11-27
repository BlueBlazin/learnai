## The idea

LayerNorm (Ba, Kiros, and Hinton, 2016) standardizes each vector independently. It subtracts the vector's mean, divides by its standard deviation, and then lets the model undo as much of that normalization as it wants with a learned gain and bias. Because nothing depends on other vectors in the batch or sequence, LayerNorm behaves identically at batch size 1 and at any sequence length, and inference needs no running statistics.

Three details separate LayerNorm from RMSNorm:

| | RMSNorm | LayerNorm |
|---|---|---|
| centering | none | subtract the mean |
| scale statistic | $\sqrt{\frac1d \sum x_j^2 + \varepsilon}$ | $\sqrt{\frac1d \sum (x_j - \mu)^2 + \varepsilon}$ |
| learned parameters | gain | gain and bias |

Centering makes the output invariant to adding a constant to the input, and dividing by the standard deviation makes it invariant to scaling (up to $\varepsilon$). RMSNorm keeps only the scaling invariance, saving one reduction and one subtraction per call.

LayerNorm uses the population variance, dividing by $d$. The layer is not estimating a statistic from a sample. The $d$ entries are the entire vector. This matches PyTorch's `torch.nn.LayerNorm`, whose documentation states that the variance uses the biased estimator (equivalent to `torch.var(input, correction=0)`) and adds $\varepsilon$ inside the square root in $y = (x - \mathrm{E}[x]) / \sqrt{\mathrm{Var}[x] + \varepsilon} \cdot \gamma + \beta$.

## Reference solution (PyTorch)

```python
def layer_norm(x, weight, bias, eps=1e-5):
    centered = x - x.mean(dim=-1, keepdim=True)
    var = centered.square().mean(dim=-1, keepdim=True)
    return centered / torch.sqrt(var + eps) * weight + bias
```

Computing the variance from `centered` means the mean is only reduced once. A constant vector yields `centered = 0` and `var = 0`, so the formula evaluates to `0 / sqrt(eps) * weight + bias = bias` without a special case because `eps` keeps the denominator positive.

The JAX and MLX versions are the same three lines with `keepdims=True`.

## Common mistakes

- **Using the unbiased variance.** `torch.var` and `torch.std` divide by $d - 1$ by default (`correction=1`), whereas `jnp.var` and `mx.var` divide by $d$ (`ddof=0`). Porting a NumPy or JAX line to PyTorch verbatim scales the output by $\sqrt{(d - 1)/d}$, which is a 13% error at $d = 4$ but only 0.07% at $d = 768$. At real model widths, that bug easily slips past a smoke test.
- **Adding `eps` outside the square root.** Writing `sqrt(var) + eps` looks identical for ordinary activations, but diverges once the variance approaches $\varepsilon$. At a variance of $3.5 \times 10^{-6}$ with $\varepsilon = 10^{-5}$, the two denominators are $3.67 \times 10^{-3}$ and $1.88 \times 10^{-3}$.
- **Omitting centering, or centering only inside the variance.** Both are RMSNorm-shaped bugs where adding a constant to the input shifts the output. Centering only inside the variance is subtler because the variance itself is right.
- **Applying bias before gain.** Computing `(normalized + bias) * weight` scales the bias by the gain. This bug is invisible with default initializations (gain 1, bias 0) and only appears with trained weights.
- **Reducing over more than the last axis.** Normalizing over `(sequence, d)` makes every token depend on other tokens in the sequence, including padding.
- **Dropping `eps`.** A constant vector, such as a padded or saturated position, produces `0 / 0` (NaN).

## Complexity

LayerNorm performs two reductions and a handful of elementwise operations along the last axis. That takes $O(d)$ time per vector and $O(d)$ extra memory for the centered copy.

## In the wild

GPT-2, BERT and its descendants (the encoders behind most embedding and reranking models), Whisper, and vision transformers all use LayerNorm. Across these models, $\varepsilon$ ranges from $10^{-12}$ (BERT) to $10^{-5}$ (GPT-2 and the PyTorch default), so always read it from the model config instead of hardcoding it. Checkpoints name the learned parameters `weight` and `bias` in PyTorch, or `gamma` and `beta` in the original TensorFlow BERT.

For large inputs with a mean far from zero, computing variance as `mean(x^2) - mean(x)^2` suffers catastrophic cancellation in float32. Computing variance directly from the centered vector avoids that loss of precision.
