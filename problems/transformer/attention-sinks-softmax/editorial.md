## The idea

Ordinary attention normalizes each row of scores to a probability distribution over
the visible keys. Attention with sinks normalizes over one more outcome, "none of
them", whose logit is a learned constant per head:

$$
a_{t,s} = \frac{e^{\text{score}_{t,s}}}{e^{\sigma} + \sum_{s'} e^{\text{score}_{t,s'}}} .
$$

The denominator is larger than the sum over the keys, so the weights on the keys sum
to $1 - p_{\text{sink}}$ with $p_{\text{sink}} = e^{\sigma} / Z$. Since the sink has
no value vector, that share simply disappears from the output: the head contributes
a shorter vector when nothing scores much higher than $\sigma$.

There are two equivalent ways to see it, and each explains one rule of the
specification.

**An extra column.** Append $\sigma$ to every row of the score matrix, take the usual
softmax over $S + 1$ entries, and delete the column. The column is appended to the
finished scores, which is why $\sigma$ is not divided by $\sqrt{d}$ and is not
subject to the mask.

**A virtual key with a zero value.** Imagine a key that scores exactly $\sigma$
against every query, with the value vector $0$. Attending to it adds nothing to the
output. This is why the weights must not be renormalized: the weight on the virtual
key is real attention, spent on a value of zero.

A row with no visible key is the limiting case. Ordinary attention has no answer for
it (the softmax of all `-inf` is `0/0`); with a sink the row is $[0, \dots, 0, 1]$
and the output is the zero vector.

Why a relative comparison and not an offset: adding $\sigma$ to every score would do
nothing, because a softmax is invariant to shifting all of its inputs. The sink
matters only as one logit among the others.

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

The JAX and MLX versions use `where` for the mask and `broadcast_to` for the column.
The framework's softmax handles `-inf` entries and large logits, so no extra care is
needed for sharp rows or for a sink of `-inf`.

## Common mistakes

- **Renormalizing.** Dividing the kept weights by their sum makes the rows sum to 1
  again and cancels the sink exactly: the result is ordinary attention. On a fully
  masked row it also divides by zero.
- **Scaling the sink.** Concatenating before the division by $\sqrt{d}$ shrinks
  $\sigma$ by that factor. With $d = 64$ a sink of 4 becomes 0.5, and the head leaks
  far less attention than it was trained to.
- **One sink for all heads**, or the sink vector broadcast along the wrong axis.
  `sinks` has shape `(H,)`. Reshaped to `(1, 1, H, 1)` it lines up with the query
  positions and runs without an error whenever `H == T`, giving query $t$ the sink of
  head $t$. Reshape to `(1, H, 1, 1)`.
- **Masking the sink.** Padding the mask with `False` for the new column, or applying
  a causal mask to the concatenated matrix, sets the sink to `-inf`.
- **Adding instead of appending.** `scores + sink` is a shift of the whole row, which
  the softmax ignores.
- **Dropping the wrong column.** If the sink is put first, `weights[..., :-1]`
  removes the last real key and multiplies the first value by the sink's share.

## Complexity

The same as ordinary attention: $O(B\,H\,T\,S\,d)$ time and $O(B\,H\,T\,S)$ memory
for the scores, plus one column.

## In the wild

The definition follows OpenAI's reference implementation of gpt-oss
(`gpt_oss/torch/model.py`, function `sdpa`): each attention block has a parameter
`sinks` with one entry per query head; the scores are multiplied by
`1 / sqrt(head_dim)` and masked (causal, plus a sliding window in every other
layer), then `QK = torch.cat([QK, S], dim=-1)`, a softmax, and `W = W[..., :-1]`
before the values are mixed. gpt-oss combines the sinks with grouped-query attention
and with sliding windows; the sink is what lets a windowed layer attend to nothing
when the window holds nothing useful. MLX exposes the same operation as the `sinks`
argument of `mx.fast.scaled_dot_product_attention`, which takes one sink per query
head and agrees with this problem's reference to float32 rounding (checked on MLX
0.32.4). The idea of giving attention somewhere harmless to go has older relatives:
the observation that streaming language models depend on the first tokens as
"attention sinks" (Xiao et al., *Efficient Streaming Language Models with Attention
Sinks*, 2023), and register tokens in vision transformers. Those keep ordinary
softmax and dedicate real tokens to the role; the learned sink logit builds the role
into the softmax itself.
