## The idea

Write the definition in matrix form for one sequence at a time. Stack the token vectors
as rows of $X \in \mathbb{R}^{T \times d}$. Then

$$
Q = X W_q^\top, \quad K = X W_k^\top, \quad V = X W_v^\top, \qquad
S = \frac{Q K^\top}{\sqrt{d_k}}, \qquad
Y = \operatorname{softmax}(S + M)\, V\, W_o^\top ,
$$

where $M_{ij} = 0$ for $j \le i$ and $-\infty$ for $j > i$, and the softmax runs over
each row. The weights are transposed because of the `(out, in)` storage layout, which
maps a row vector $x^\top$ to $x^\top W^\top$.

Three details matter most when implementing these equations.

**Apply the mask to the logits.** Adding $-\infty$ before softmax removes future
positions from both the numerator and the denominator. Row $i$ becomes a valid
probability distribution over positions $0 \dots i$ that depends only on $q_i$ and
$k_0 \dots k_i$, which guarantees causality by construction.

**Scale by $1/\sqrt{d_k}$.** The dot product of two $d_k$-dimensional vectors with
unit-variance entries has variance $d_k$. Dividing by $\sqrt{d_k}$ keeps the logits at
unit scale regardless of head width, which trained model weights expect.

**Keep the batch axis untouched.** Every step is a batched matrix product or a reduction
over the last axis. `k.transpose(-2, -1)` swaps only the sequence and feature axes of
`k`, and the `(T, T)` mask broadcasts across the batch dimension.

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

JAX uses `jnp.where(future, -jnp.inf, scores)` and MLX uses `mx.where`, since neither
supports in-place masking. The rest of the implementation is identical. Because row 0
always has one visible entry on the diagonal, no row is entirely `-inf` and softmax
never evaluates `0 / 0`.

## Common mistakes

- **Mask in the wrong direction.** `tril` and `triu` are easy to mix up, as are masks of
  what to keep versus what to hide. With a flipped mask, every token sees the future
  instead of the past. Check row 0 to make sure it attends only to itself.
- **Masking after the softmax.** Zeroing future weights after softmax produces the right
  shape and zeros in the right places, but the visible weights were divided by a
  denominator that included future tokens. Rows no longer sum to 1, and earlier outputs
  change when later tokens change.
- **Setting masked scores to 0 instead of $-\infty$.** A logit of 0 does not mean zero
  attention because $e^0 = 1$ still contributes weight.
- **Masking the diagonal too.** Calling `triu` without `diagonal=1` hides $j \ge i$.
  Row 0 becomes entirely $-\infty$, and its softmax produces NaN.
- **Missing or wrong scale.** Forgetting $1/\sqrt{d_k}$ or dividing by $\sqrt{d}$ (the
  width of `x`) instead of $\sqrt{d_k}$ breaks the logits' scale. Using $\sqrt{d}$ is
  invisible whenever $d_k = d$, which happens in single-head examples but never in
  multi-head models.
- **Swapping queries and keys.** Computing `k @ q^T` transposes the score matrix. If you
  apply the causal mask afterward, the output is still causal, so only a numeric check
  catches it.
- **Not transposing the weights.** Writing `x @ w_q` runs without error when the
  projection is square and silently applies the transposed linear map.
- **Leaking across the batch.** Flattening `(B, T)` into one axis, or transposing axis 0
  with another axis, lets sequences attend to one another. Testing only with `B = 1`
  hides this bug.

## Complexity

Computing the scores takes $O(B\,T^2\,d_k)$ time and $O(B\,T^2)$ memory for the
attention matrix. The linear projections take $O(B\,T\,d\,(d_k + d_v))$ time. The
quadratic $T^2$ term is why long contexts are expensive.

## In the wild

Inference engines do not run this full matrix form during token-by-token generation.
With a KV cache, each decoding step computes only one new row of scores by multiplying
the new token's query against all cached keys. No mask is needed for that row because
every cached key is in the past. That cache works only because causality guarantees
that appending a token never changes earlier keys, values, or outputs.

For full-sequence passes during prompt processing or training, frameworks use fused
kernels such as PyTorch's `scaled_dot_product_attention(..., is_causal=True)` that
never materialize the $T \times T$ matrix in memory. Multi-head attention runs this same
operation in parallel across $H$ slices of width $d / H$, which is why $d_k = d / H$
and why the scale uses $d_k$ rather than $d$. Production masks are usually additive
tensors (adding a large negative float to masked logits) that combine the causal mask
with a padding mask so sequences of different lengths can share a batch.
