## The idea

Write the definition in matrix form, one sequence at a time. Stack the token vectors as
rows of $X \in \mathbb{R}^{T \times d}$. Then

$$
Q = X W_q^\top, \quad K = X W_k^\top, \quad V = X W_v^\top, \qquad
S = \frac{Q K^\top}{\sqrt{d_k}}, \qquad
Y = \operatorname{softmax}(S + M)\, V\, W_o^\top ,
$$

where $M_{ij} = 0$ for $j \le i$ and $-\infty$ for $j > i$, and the softmax is over each
row. The transposes on the weights come from the `(out, in)` storage layout: a row
vector $x^\top$ maps to $x^\top W^\top$.

Three details carry all of the difficulty.

**The mask goes on the logits.** Adding $-\infty$ before the softmax removes the hidden
positions from both the numerator and the denominator, so row $i$ is a proper
distribution over positions $0..i$ that does not depend on anything after $i$. That is
the causality property, and it holds by construction: row $i$ of $S + M$ contains only
$q_i$ and $k_0..k_i$.

**The scale is $1/\sqrt{d_k}$.** A dot product of two $d_k$-dimensional vectors with
unit-variance entries has variance $d_k$; dividing by $\sqrt{d_k}$ keeps the logits at
unit scale whatever the head width. A trained model's weights assume it.

**The batch axis is never touched.** Every operation is a batched matrix product or a
reduction over the last axis. `k.transpose(-2, -1)` swaps key position and feature;
the mask has shape `(T, T)` and broadcasts.

## Reference solution (PyTorch)

```python
def causal_self_attention(x, w_q, w_k, w_v, w_o):
    seq = x.shape[1]
    q, k, v = x @ w_q.T, x @ w_k.T, x @ w_v.T
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return (attn @ v) @ w_o.T
```

JAX uses `jnp.where(future, -jnp.inf, scores)` and MLX `mx.where`, since neither has
in-place masking; the rest is identical. Row 0 always has one visible entry (the
diagonal), so no row is entirely `-inf` and the softmax never sees `0 / 0`.

## Common mistakes

- **Mask in the wrong direction.** `tril` and `triu` are easy to confuse, and so is
  "mask of what to keep" versus "mask of what to hide". With the mask flipped, every
  token sees the future and not the past. Check row 0: it must attend to itself only.
- **Masking after the softmax.** The future weights become zero and the shape is right.
  But the visible weights were divided
  by a sum that includes the future, so rows no longer sum to 1 and earlier outputs
  change when later tokens change.
- **Setting masked scores to 0 instead of $-\infty$.** A logit of 0 is not "no
  attention"; $e^0 = 1$ is an ordinary weight.
- **Masking the diagonal too.** `triu` without the offset hides $j \ge i$. Row 0 is
  then entirely $-\infty$ and its softmax is NaN.
- **Missing or wrong scale.** Forgetting $1/\sqrt{d_k}$, or dividing by $\sqrt{d}$ (the
  width of `x`). The second is invisible whenever $d_k = d$, which is exactly the case
  in a single-head toy and never the case in a multi-head model.
- **Swapping queries and keys.** `k @ q^T` is the transpose of the score matrix. With
  the causal mask applied afterwards it is still causal, so only a numeric comparison
  catches it.
- **Not transposing the weights.** `x @ w_q` works without an error for square
  projections and silently applies the transposed map.
- **Leaking across the batch.** Flattening `(B, T)` into one axis, or transposing axis
  0 with another, lets one sequence attend to another. A batch of one hides it.

## Complexity

$O(B\,T^2\,d_k)$ time for the scores and $O(B\,T^2)$ memory for the attention matrix,
plus $O(B\,T\,d\,(d_k + d_v))$ for the projections. The quadratic term in $T$ is why
long contexts are expensive.

## In the wild

Engines never materialize this function in this form during generation. With a KV
cache, each decoding step computes one new row of the score matrix: the new token's
query against all cached keys. No mask is needed for that row, because everything in
the cache is in the past. The cache is only correct because of the causality property
you just tested: appending a token cannot change earlier keys, values, or outputs.

For the full-sequence pass (the prompt, or training), frameworks provide fused kernels
such as PyTorch's `scaled_dot_product_attention(..., is_causal=True)` that never build
the $T \times T$ matrix in memory. Multi-head attention is this function applied to
$H$ slices of width $d / H$ in parallel, which is why $d_k = d / H$ and why the scale
uses $d_k$ and not $d$. Real masks are usually additive (a large negative float added
to the scores) and combine the causal mask with a padding mask, so that a batch of
sequences of different lengths can share one tensor.
