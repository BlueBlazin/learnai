## The idea

An encoder outputs a vector for every position in a batch of shape `(B, T, D)`, including padded positions whose values are arbitrary. Pooling must aggregate only the real tokens indicated by `mask`.

**Mean pooling.** Let $s_{b,t}$ be the token state at sequence $b$ and position $t$, and let $m_{b,t} \in \{0, 1\}$ be the corresponding mask value. Multiply each state by its mask so padded positions contribute zero to the sum, then divide by the number of real tokens in that sequence:

$$
p_b = \frac{\sum_t m_{b,t}\, s_{b,t}}{\sum_t m_{b,t}} .
$$

**First and last token pooling.** Both modes select one position per sequence, and that index depends on the padding side. For a sequence with $L$ real tokens, right padding places the first real token at index `0` and the last at `L - 1`, whereas left padding places them at `T - L` and `T - 1`.

You can avoid branching on the padding side altogether. The last real token is always the largest position $t$ with $m_{b,t} = 1$, and the first real token is always the smallest. Multiply `arange(T)` by the mask so every padded position becomes `0`, which cannot beat any real position in a `max` reduction (and if position `0` is the only real token, `0` is the right answer). For the first real token, replace padded positions with `T` so they cannot win a `min` reduction. A single gather then extracts the chosen token state for every sequence.

**Normalization.** Cosine similarity equals the dot product when both vectors have unit length, so search indexes store normalized embeddings. Clamping the L2 norm from below at $10^{-12}$ leaves every nonzero vector unchanged while preventing an all-zero vector from dividing by zero and producing NaN.

**Matryoshka truncation.** Matryoshka Representation Learning trains a model with the same loss applied to multiple vector prefixes (such as the first 64, 128, or 256 dimensions), making each prefix a standalone embedding. Slicing a unit vector to its first `dim` coordinates reduces its norm below 1 by an amount that varies from text to text. Renormalizing after slicing restores unit length so inner-product search matches cosine similarity again.

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

The JAX and MLX implementations follow the same structure, using `jnp.where(...).min(axis=1)` or `mx.maximum(norm, EPS)` as appropriate.

## Common mistakes

- **Averaging over padded positions (`states.mean(dim=1)`).** Each text's embedding then shifts based on the longest sequence in its batch.
- **Using `states[:, -1]` for `"last"`.** This works for left-padded batches (common in decoder models) but reads a padding state for every shorter sequence in a right-padded batch.
- **Using `lengths - 1` for `"last"`.** This works for right-padded batches but points into padding or an earlier token when the batch is left-padded.
- **Using `states[:, 0]` for `"cls"`.** This works for right-padded batches where encoder models place `[CLS]` at index 0, but reads padding in left-padded sequences.
- **Omitting the norm floor.** An all-zero pooled vector divides `0 / 0` and turns into NaN.
- **Normalizing along the wrong axis.** Calling `norm(dim=0)` normalizes each feature across the batch instead of each embedding vector. When `B == D`, the shapes still match and no error is raised.
- **Skipping normalization in `pool` or `truncate_matryoshka`.** Inner-product search over vectors of unequal lengths conflates magnitude with direction.

## Complexity

Mean pooling runs in $O(BTD)$ time and allocates a `(B, T, D)` masked tensor. First-token and last-token pooling run in $O(BT + BD)$ time. Matryoshka truncation runs in $O(B \cdot \texttt{dim})$ time.

## In the wild

- Sentence Transformers implements these operations as a `Pooling` module (supporting `mean`, `cls`, `lasttoken`, and other modes) followed by a `Normalize` module.
- The E5 model cards ship an `average_pool` helper that masks padded states, divides by the mask sum, and applies `F.normalize`, matching the `"mean"` mode here.
- The Qwen3-Embedding model card ships a `last_token_pool` helper that checks whether every row's final column is a real token. If so, it takes `last_hidden_states[:, -1]`. Otherwise, it gathers at `attention_mask.sum(dim=1) - 1`. Its tokenizer defaults to `padding_side="left"`. The `max`-over-positions formula handles both cases without branching.
- Matryoshka Representation Learning (arXiv:2205.13147) introduced nested prefix training. Models such as Qwen3-Embedding support truncating outputs to any dimension from 32 up to the full embedding size.
