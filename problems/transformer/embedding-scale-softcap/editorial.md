## The idea

A decoder-only model is almost entirely described by its tensors, and what is left
over lives in a few scalars of `config.json`. Three of them sit at the two ends of
the forward pass.

**The embedding multiplier.** Embedding rows are small vectors, while the residual
stream they start is added to by every block. Some families scale the embeddings up so that
the token identity is not drowned out: the original Transformer and Gemma multiply by
$\sqrt{d}$, Granite by a tuned constant (12 for Granite 4.0 350M). Llama-style models
use 1. The factor belongs to the *use* of the matrix as an embedding, not to the
matrix: with tied weights the same $E$ is the output head, and there it is used as
stored.

**The logit divisor.** Granite divides its logits by `logits_scaling` (4 for the
350M model). Dividing logits by a constant is a temperature: it does not change the
argmax, and it changes every probability. A port that omits it samples from the model
at temperature 0.25.

**The soft cap.** Gemma 2 passes its final logits through
$c \tanh(z / c)$ with $c = 30$ (and its attention scores through the same function
with $c = 50$). The function is the identity to first order,
$c\tanh(z/c) = z - z^3 / (3c^2) + \dots$, and saturates at $\pm c$. Unlike a clip it
is strictly increasing, so it never creates ties between large logits and keeps a
gradient everywhere, which is why it is used in training and therefore has to be
reproduced at inference.

**The order.** No current model has both a divisor and a cap, so the order is this
problem's convention: divide, then cap. It is the natural one (the cap bounds what
the sampler will see), and the two orders differ:
$c\tanh\!\big(\tfrac{z}{sc}\big) \ne \tfrac{c}{s}\tanh\!\big(\tfrac{z}{c}\big)$; the
second can never exceed $c / s$.

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

The JAX and MLX versions differ only in the name of `tanh`. `tanh` saturates without
overflowing, so nothing special is needed for logits far beyond the cap.

## Common mistakes

- **Scaling the table.** `table *= scale` (or folding the multiplier into the
  weights when loading) gives the right embeddings and a head that is `scale` times
  too large in every tied model. The bug is invisible to a test of `embed` alone; it
  shows when the same tensor is used for both.
- **Capping before dividing.** With a divisor of 4 and a cap of 8, raw logits of 40
  should become $8\tanh(10/8) = 6.8$; capping first gives $8\tanh(5)/4 = 2.0$.
- **A hard clip.** `clamp(z, -c, c)` agrees with the soft cap for small logits and
  at the extremes, and is off by up to 24 percent of the cap in between (at $z = c$,
  $c\tanh 1 = 0.76c$).
- **Forgetting the inner division.** `c * tanh(z)` saturates at $|z|$ of about 3
  instead of about $3c$: almost every logit comes out as $\pm c$.
- **Multiplying by the logit scale.** The config name `logits_scaling` does not say
  which way it goes. In Granite it is a divisor.
- **Hard-coding a family's constant.** `x * d ** 0.5` is Gemma's embedding scale and
  is wrong for every model whose config says otherwise. Take the number from the
  config.

## Complexity

The lookup is $O(Td)$ for $T$ tokens. The head is one matrix product, $O(T d V)$ for
$T$ positions, which is why engines compute it for the last position only during
generation (see `last-position-head`); the division and the cap are $O(TV)$.

## In the wild

Hugging Face Transformers: `GraniteMoeHybridModel.forward` computes
`inputs_embeds * self.embedding_multiplier` and the causal-LM head computes
`logits / self.config.logits_scaling`; Gemma 2 scales its embeddings by
`hidden_size ** 0.5` (the constant is cast to the dtype of the embedding weights
first, so in bfloat16 the factor is the nearest bfloat16 value to $\sqrt{d}$) and
applies `final_logit_softcapping` as divide, `tanh`, multiply, exactly the formula
above. `real-model-logits` in this track verifies the two Granite constants against
the real checkpoint.
