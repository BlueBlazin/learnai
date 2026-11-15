## The idea

An encoder gives every position a state, including the padded positions, whose
states are meaningless. Pooling must read the real positions only, and the mask is
the only thing that says which those are.

**Mean.** Multiply by the mask so that padded states contribute zero to the sum,
and divide by the count of real tokens:

$$
p_b = \frac{\sum_t m_{b,t}\, s_{b,t}}{\sum_t m_{b,t}} .
$$

**First and last token.** Both are "pick one position per row", and the position
is not a constant. With right padding the first real token is at position 0 and
the last at `lengths - 1`; with left padding the first is at `T - lengths` and the
last at `T - 1`. Formulas that case-split on the padding side work, but there is
one that does not need to know the side: the last real position is the largest $t$
with $m_{b,t} = 1$, and the first is the smallest. Multiplying `arange(T)` by the
mask makes every padded position compete with the value 0, which cannot beat a
real position in a `max`; replacing padded positions by `T` does the same for the
`min`. One gather then selects the state.

**Normalization.** Cosine similarity is an inner product of unit vectors, so
indexes store unit vectors and the model's output is divided by its length. The
floor $10^{-12}$ on the divisor only matters for the zero vector, which would
otherwise become NaN and poison every score it takes part in.

**Matryoshka truncation.** Matryoshka Representation Learning trains a model with
the same loss applied to several prefixes of the embedding (the first 64, 128, 256,
... coordinates), so each prefix is a usable embedding on its own. The model
outputs a unit vector of the full size; a prefix of it has length below 1, and how
far below differs from text to text. Renormalizing after slicing puts the short
vectors back on the unit sphere, where cosine and inner product agree.

## Reference solution (PyTorch)

```python
EPS = 1e-12


def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def pool(states, mask, mode):
    batch, length, _ = states.shape
    positions = torch.arange(length)
    if mode == "mean":
        weights = mask.to(states.dtype).unsqueeze(-1)
        pooled = (states * weights).sum(dim=1) / weights.sum(dim=1)
    else:
        if mode == "cls":
            index = torch.where(mask != 0, positions, length).amin(dim=1)
        else:
            index = (positions * (mask != 0)).amax(dim=1)
        pooled = states[torch.arange(batch), index]
    return _normalize(pooled)


def truncate_matryoshka(e, dim):
    return _normalize(e[..., :dim])
```

The JAX and MLX versions are line-for-line translations (`jnp.where(...).min(axis=1)`,
`mx.maximum(norm, EPS)`).

## Common mistakes

- **Averaging over padding** (`states.mean(dim=1)`). The embedding of a text then
  depends on how long the longest text in its batch was.
- **`states[:, -1]` for the last token.** Right for left-padded batches, which is
  how decoder models are usually batched, and a padding state for every shorter
  row of a right-padded batch.
- **`lengths - 1` for the last token.** The mirror image: right for right padding,
  and with left padding it points into the padding or at an early token.
- **`states[:, 0]` for the first token.** Right for right padding, where
  encoder-style models put their `[CLS]` token; padding for left-padded rows.
- **No floor on the norm.** An all-zero pooled vector becomes NaN.
- **Normalizing along the wrong axis.** `norm(dim=0)` normalizes each coordinate
  across the batch. With `B = D` the shapes agree and nothing complains.
- **Forgetting to normalize**, in `pool` or after truncating. Inner-product search
  over vectors of unequal length ranks by length as much as by direction.

## Complexity

$O(BTD)$ for the mean and $O(BT + BD)$ for the token poolings; nothing is
allocated beyond the `(B, T, D)` product for the mean. Truncation is $O(B \cdot
\texttt{dim})$.

## In the wild

- Sentence Transformers implements these as its `Pooling` module (modes `mean`,
  `cls`, `lasttoken`, and others) followed by a `Normalize` module.
- The E5 model cards ship an `average_pool` function that zeroes the padded states
  with the attention mask and divides by the mask's sum, followed by
  `F.normalize`: the `"mean"` mode here.
- The Qwen3-Embedding model card ships `last_token_pool`: if every row's last
  column is a real token the batch is left-padded and it returns
  `last_hidden_states[:, -1]`; otherwise it gathers at `attention_mask.sum(dim=1)
  - 1`. Its tokenizer is configured with `padding_side="left"`. That is the
  case split the `max`-over-positions formula replaces.
- Matryoshka Representation Learning (arXiv:2205.13147) introduced the nested
  training; the Qwen3-Embedding cards, for example, state support for output
  dimensions from 32 up to the model's full size.
