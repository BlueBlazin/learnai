## The idea

Standard attention normalizes each row of key scores into a probability distribution over the visible keys. Attention with sinks adds one extra outcome, "none of the keys", whose logit is a learned constant $\sigma$ per head:

$$
a_{t,s} = \frac{e^{\text{score}_{t,s}}}{e^{\sigma} + \sum_{s'} e^{\text{score}_{t,s'}}} .
$$

Because the denominator $Z = e^{\sigma} + \sum_{s'} e^{\text{score}_{t,s'}}$ includes the sink term, the weights on the real keys sum to $1 - p_{\text{sink}}$, where $p_{\text{sink}} = e^{\sigma} / Z$. The sink has no value vector, so its probability share drops out of the output. When no key scores much higher than $\sigma$, the head simply outputs a shorter vector.

Two equivalent interpretations explain every rule in the specification:

- **An extra column.** Append $\sigma$ to the end of each row in the score matrix, run the usual softmax over all $S + 1$ entries, and slice off the last column. Because the column is appended to the finished scores, $\sigma$ is neither divided by $\sqrt{d}$ nor affected by the mask.
- **A virtual key with a zero value.** Picture a key that always scores $\sigma$ against every query and carries the zero value vector. Attending to this virtual key contributes zero to the sum. That is why the remaining key weights must not be renormalized, since the missing probability was spent on a zero value.

A fully masked row is the limiting case. Standard attention produces `0/0` (NaN) on a row of all `-inf`. With a finite sink, the softmax over $[-\infty, \dots, -\infty, \sigma]$ is $[0, \dots, 0, 1]$, and slicing off the sink leaves a zero output vector.

Note that adding $\sigma$ to every score would have no effect, because softmax is invariant to constant shifts across a row. The sink only works because it competes as a separate logit against the key scores.

## Reference solution (PyTorch)

```python
def attention_with_sinks(q, k, v, mask, sinks):
    batch, n_heads, n_queries, _ = q.shape
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, S)
    scores = scores.masked_fill(~mask, float("-inf"))
    sink = sinks.reshape(1, n_heads, 1, 1).expand(batch, n_heads, n_queries, 1)
    weights = torch.softmax(torch.cat((scores, sink), dim=-1), dim=-1)  # (B, H, T, S + 1)
    return weights[..., :-1] @ v
```

The JAX and MLX versions use `where` for masking and `broadcast_to` to expand the sink column. Built-in `softmax` implementations already handle `-inf` entries and large logits safely, so sharp distributions and `sinks[h] = -inf` require no special casing.

## Common mistakes

- **Renormalizing the kept weights.** Dividing the key weights by their sum forces each row back to 1 and cancels the sink completely, reverting to standard attention. On a fully masked row, it also divides by zero.
- **Scaling the sink by $1/\sqrt{d}$.** Concatenating the sink before scaling shrinks $\sigma$ by $\sqrt{d}$. For $d = 64$, a sink of 4 becomes 0.5, causing the head to absorb far less attention than trained.
- **Broadcasting `sinks` along the wrong axis.** `sinks` has shape `(H,)`. Reshaping it to `(1, 1, H, 1)` aligns it with the query axis `T` instead of the head axis `H`. Whenever `H == T`, this runs without a shape error and silently gives query $t$ the sink of head $t$. Always reshape to `(1, H, 1, 1)`.
- **Masking the sink.** Padding `mask` with `False` for the extra column, or applying the mask after concatenation, sets the sink logit to `-inf` and disables it.
- **Adding instead of concatenating.** Computing `scores + sink` shifts the entire row by a constant, which softmax ignores.
- **Dropping the wrong column.** If you prepend the sink at index 0, slicing `weights[..., :-1]` drops the last real key and multiplies the first value vector by the sink weight.

## Complexity

Attention with sinks has the same asymptotic cost as standard attention: $O(B\,H\,T\,S\,d)$ time and $O(B\,H\,T\,S)$ memory for the score matrix, plus a single extra column.

## In the wild

This specification matches OpenAI's reference implementation of gpt-oss (`gpt_oss/torch/model.py`, function `sdpa`). Each attention block holds a `sinks` parameter with one logit per query head. After scaling scores by `1 / sqrt(head_dim)` and applying causal and sliding-window masks, the code runs `QK = torch.cat([QK, S], dim=-1)`, applies softmax, and slices `W = W[..., :-1]` before multiplying by `V`. In gpt-oss, learned sinks pair with grouped-query attention and sliding windows, allowing a windowed layer to attend to nothing when the local window holds no useful context.

MLX exposes the same operation natively through the `sinks` argument of `mx.fast.scaled_dot_product_attention`, which accepts one sink per query head and agrees with this reference to float32 precision (verified on MLX 0.32.4). Earlier work achieved a similar effect using real tokens rather than modifying softmax, including streaming "attention sinks" on initial tokens (Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023) and register tokens in vision transformers. Learned sink logits build that escape hatch directly into the softmax.
