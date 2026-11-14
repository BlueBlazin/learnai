## The idea

A constraint does not change what the model computes. It changes which outcomes the
sampler may pick. The distribution you want is the model's distribution *conditioned
on* the next token being allowed:

$$
P(v \mid v \in A) = \frac{p_v}{\sum_{w \in A} p_w}
= \frac{e^{\ell_v}}{\sum_{w \in A} e^{\ell_w}} \quad (v \in A).
$$

The normalizer of the full softmax cancels, so the conditioned distribution is simply
the softmax over the allowed logits. Writing `-inf` into the disallowed positions
expresses exactly that: $e^{-\infty} = 0$, so those tokens contribute nothing to the
sum and receive nothing from it. `-inf` is also the only value that stays removed
under everything a sampler does afterward: dividing by a temperature, adding a logit
bias, or filtering by top-p.

The order of the two operations matters numerically. The stable softmax subtracts the
row maximum before exponentiating. If the mask is applied first, that maximum is the
largest allowed logit and the allowed tokens are computed at full precision. If the
softmax runs first, the maximum may belong to a disallowed token, and allowed tokens
that are more than about 100 below it underflow to zero in float32. Renormalizing then
divides zero by zero. This is not an exotic input: a model that badly wants to write
prose when the grammar demands `{` produces exactly this row.

A row with no allowed token has no conditioned distribution. Its masked logits are all
`-inf`, the softmax of that is NaN, and NaN reaches the sampler as an out-of-range
index or a crash several layers away from the cause. So the check belongs here, per
row, before anything is computed.

## Reference solution (PyTorch)

```python
def mask_logits(logits, allowed):
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("a row of `allowed` has no allowed token")
    return torch.where(allowed, logits, float("-inf"))


def masked_probs(logits, allowed):
    return torch.softmax(mask_logits(logits, allowed), dim=-1)
```

The JAX and MLX versions are the same lines with `jnp.where` / `mx.where` and their
`softmax`. In JAX the check reads a concrete value (`bool(...)`), which is fine here
because the function is not traced; inside `jax.jit` you would return a per-row "dead"
flag instead and let the caller raise.

## Common mistakes

- **A large negative constant.** `-1e9` gives the right probabilities today and breaks
  when something downstream touches the logits: an added bias or a second mask shifts
  it, half precision cannot represent it, and code that looks for masked tokens with
  `isinf` no longer finds them. With `-inf` there is nothing to tune.
- **Multiplying probabilities by the mask without renormalizing.** The row then sums
  to the allowed mass, not to 1. A sampler that draws `u` in `[0, 1)` against the
  cumulative sum runs off the end of the row whenever `u` exceeds that mass.
- **Softmax first, mask and renormalize second.** Mathematically the same, and NaN as
  soon as the allowed mass underflows (see above).
- **Not checking for an empty row**, which returns NaN, or checking `allowed.any()`
  over the whole batch, which only fires when *every* row is dead. In a batched
  server one request's dead end must not go unnoticed because its neighbors are fine.
- **Sharing one mask across the batch.** Each sequence is in its own constraint state.
  A mask of shape `(V,)` broadcast over the batch is right only when all rows happen
  to be at the same point of the same grammar.
- **An additive mask built by multiplication**, `logits + (~allowed) * -inf`: the
  product is `0 * -inf = NaN` at every allowed position.

## Complexity

One pass over the `B * V` entries for the selection and one for the softmax: $O(BV)$
time and memory. In practice the cost of constrained decoding is in *computing*
`allowed`, which the rest of this track is about, and in moving it to the device: a
boolean per token for a 128,000-token vocabulary is 128 kB per sequence per step,
which is why engines pack the mask into bits.

## In the wild

- Hugging Face `transformers` implements every constraint as a logits processor that
  writes `-inf` into the scores of disallowed tokens before sampling; top-k and top-p
  use the same convention, which is why the filters compose.
- llama.cpp's grammar sampler sets the logit of each rejected candidate to
  `-INFINITY`.
- XGrammar and llguidance return the mask as a packed bitset, 32 tokens per integer,
  and provide a routine that applies it to the logits in place. vLLM collects one
  bitmask row per constrained request of a batch and applies them to those rows of
  the logits in a single call.
- Where the mask sits in the sampler chain matters. Applied before top-k and top-p,
  those filters choose among allowed tokens. Applied after them, the top-k set may
  contain no allowed token at all, which is the empty row this problem makes you
  detect.
