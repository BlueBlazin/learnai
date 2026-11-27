## The idea

Most of a decoder-only model is defined by its weight tensors, while a few remaining details live as scalar constants in `config.json`. Three of those scalars sit at the input and output boundaries of the forward pass.

**The embedding multiplier.** Raw embedding rows have small norms, while every Transformer block adds to the residual stream. Several model families scale embeddings up so the initial token signal is not drowned out. The original Transformer and Gemma multiply by $\sqrt{d}$, Granite multiplies by a tuned constant (12 for Granite 4.0 350M), and Llama-style models use 1. This multiplier belongs to the embedding lookup rather than the weight matrix itself. When embeddings and the output head are tied, the output head uses the stored matrix $E$ without scaling.

**The logit divisor.** Granite divides its output logits by `logits_scaling` (4 for the 350M model). Dividing logits by a constant acts like temperature scaling. It leaves the argmax unchanged while altering every token probability, so a port that omits it effectively samples at temperature 0.25.

**The soft cap.** Gemma 2 passes its final logits through $c \tanh(z / c)$ with $c = 30$ (and applies the same function to attention scores with $c = 50$). Near zero, the function behaves like the identity, $c\tanh(z/c) = z - z^3 / (3c^2) + \dots$, and for large inputs it saturates at $\pm c$. Unlike a hard clip, it is strictly increasing, so it never creates ties among large logits and keeps a nonzero gradient everywhere. Because the model is trained with this cap, inference must apply it as well.

**The order of operations.** No current model uses both a logit divisor and a soft cap, so dividing before capping is this problem's convention. It is the natural order because the cap bounds the logits that reach the sampler. With $s = \texttt{logit\_scale}$, the two orders are not equivalent: $c\tanh\!\big(\tfrac{z}{sc}\big) \ne \tfrac{c}{s}\tanh\!\big(\tfrac{z}{c}\big)$, and capping first would restrict outputs to $\pm c / s$.

## Reference solution (PyTorch)

```python
def embed(tokens, table, scale):
    return table[tokens] * scale


def finalize_logits(h, w_out, logit_scale, softcap=None):
    logits = h @ w_out.T / logit_scale
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    return logits
```

The JAX and MLX solutions differ only in the `tanh` function name. Because `tanh` saturates cleanly to $\pm 1$ without overflowing, no extra clamping is needed for logits far larger than `softcap`.

## Common mistakes

- **Mutating the embedding table.** Running `table *= scale` or folding the multiplier into the weights at load time produces correct embeddings, but makes the output head `scale` times too large in any tied-weight model. Testing `embed` alone misses this bug until the same tensor is passed to `finalize_logits`.
- **Capping before dividing.** With a divisor of 4 and a cap of 8, a raw logit of 40 should become $8\tanh(10/8) = 6.8$. Capping first gives $8\tanh(5)/4 = 2.0$.
- **Using a hard clip.** `clamp(z, -c, c)` matches the soft cap near zero and at infinity, but is off by up to 24% of `c` in between. At $z = c$, $c\tanh(1) \approx 0.76c$.
- **Forgetting the inner division.** Writing `c * tanh(z)` saturates around $|z| \approx 3$ instead of $|z| \approx 3c$, pushing almost every logit to $\pm c$.
- **Multiplying by `logit_scale`.** The config key `logits_scaling` does not specify direction. In Granite, it is a divisor.
- **Hard-coding a model family's constant.** Using `x * d ** 0.5` works for Gemma but fails for any model whose config specifies a different scale. Always read the constant from the argument.

## Complexity

The embedding lookup costs $O(Td)$ for $T$ tokens. The output projection is one matrix multiplication costing $O(T d V)$, which is why inference engines run it only on the last position during generation (see `last-position-head`). The division and soft cap are elementwise and cost $O(TV)$.

## In the wild

In Hugging Face Transformers, `GraniteMoeHybridModel.forward` computes `inputs_embeds * self.embedding_multiplier`, and the causal-LM head computes `logits / self.config.logits_scaling`. Gemma 2 scales embeddings by `hidden_size ** 0.5`, casting that constant to the embedding weight dtype first so that in bfloat16 it uses the nearest bfloat16 value to $\sqrt{d}$. It then applies `final_logit_softcapping` by dividing, taking `tanh`, and multiplying, matching the formula here. `real-model-logits` in this track checks Granite's constants against the real checkpoint.
