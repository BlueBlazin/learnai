## The idea

A residual stream is one vector per token that every sublayer reads and adds to.
Hyper-connections widen it to $N$ vectors and make both the reading and the adding
learned and input-dependent:

- **read**: the sublayer's input is $\sum_n \texttt{pre}_n X_n$;
- **write**: the sublayer's output $y$ is added to stream $n$ with weight
  $\texttt{post}_n$;
- **carry**: the old streams are not copied across as in $x + f(x)$ but mixed,
  $X' = \texttt{comb}^\top X$.

With $N = 1$, $\texttt{pre} = \texttt{post} = \texttt{comb} = 1$ this is the ordinary
residual connection. All three weights come from one linear map of the normalized,
flattened streams, so each token chooses its own routing.

The carry term is the dangerous one: it is applied at every sublayer, so the product
of all the mixing matrices sits between the embedding and the output. An unconstrained
matrix would let that product grow or vanish with depth. The "manifold-constrained"
variant pushes each `comb` toward the doubly stochastic matrices (non-negative entries,
every row and column summing to 1) with Sinkhorn-Knopp: alternately rescale columns
and rows until both sum to 1.

## What the iteration guarantees, and what it only approaches

The last step is always a column step, and the merge uses `comb` transposed, so column
$n$ holds the weights of new stream $n$. Together these give an exact statement for
any iteration count: **every new stream is a weighted average of the old ones**,
weights positive and summing to $s / (s + \texttt{eps})$ with $s$ the column sum before
the last step. The carry term cannot increase the largest stream norm.

Row sums are different. A row sum says how much of old stream $j$ survives in total,
and it is 1 only in the limit. How close 20 iterations get depends on how peaked the
softmax was. Measured on 200,000 random and 200,000 two-valued $4 \times 4$ matrices of
softmax inputs spanning a given range, the largest row-sum error after 20 iterations
(float64, `eps = 1e-6`):

| span of the softmax inputs | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| largest row-sum error | 1e-6 | 9e-4 | 6e-3 | 2e-2 | 3e-2 |

So "doubly stochastic" is exact for columns and approximate for rows, and for sharp
matrices 20 iterations are far from converged. Two consequences for an implementation:

- Choices that only change the starting point of the iteration, such as starting with
  rows, taking the softmax over the other axis, or running one extra round, give the
  same matrix once it has converged, up to the `eps` terms (all are diagonal rescalings
  of the same positive matrix, and the doubly stochastic rescaling is unique). They
  are wrong all the same: at 1 or 2 iterations, and at 20 for sharp matrices, the
  results differ in the first or second digit. The tests use 1, 2, and 3 iterations to pin the order down.
- Ending on rows instead of columns moves the exact guarantee to the wrong side:
  after the transpose in the merge, the new streams are then no longer averages.

## Reference solution (PyTorch)

```python
def hc_weights(streams, fn, base, scale, iters, eps):
    steps, n, dim = streams.shape
    flat = streams.reshape(steps, n * dim)
    flat = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + 1e-6)
    raw = flat @ fn.T
    pre = torch.sigmoid(raw[:, :n] * scale[0] + base[:n]) + eps
    post = 2.0 * torch.sigmoid(raw[:, n : 2 * n] * scale[1] + base[n : 2 * n])
    logits = (raw[:, 2 * n :] * scale[2] + base[2 * n :]).reshape(steps, n, n)
    comb = torch.softmax(logits, dim=-1) + eps

    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    collapsed = (pre[:, :, None] * streams).sum(dim=1)
    return collapsed, post, comb


def hc_merge(streams, y, post, comb):
    return post[:, :, None] * y[:, None, :] + comb.transpose(-1, -2) @ streams
```

## Common mistakes

- **`comb @ streams`.** Without the transpose, new stream $n$ is weighted by row $n$,
  whose sum is only approximately 1. For a symmetric matrix nothing changes, and every
  $2 \times 2$ doubly stochastic matrix is symmetric, so a converged two-stream test
  cannot see it.
- **No Sinkhorn step,** or rows first, or one round too many, or the softmax over
  columns: see above.
- **`post` without the factor 2.** The sigmoid's midpoint is then a weight of 0.5, and
  a freshly initialized model writes half of every sublayer output.
- **Averaging the streams** instead of weighting by `pre`.
- **A norm per stream.** Normalizing each stream over its own $D$ channels makes all
  streams the same size before the projection; the model normalizes the flattened
  vector once, which keeps their ratios.
- **The wrong epsilon.** `eps` is added to `pre` (so no stream is ever switched off
  completely), to the softmax (so no entry of `comb` is exactly zero, which Sinkhorn
  could never repair), and to each divisor. It is not added to `post`, and it is not
  the epsilon of the norm, which is `rms_norm_eps`. Both are 1e-6 in the shipped
  configuration, which is how the confusion goes unnoticed; the tests use `eps = 0.01`.
- **`(raw + base) * scale`.** The scale multiplies the projected value only.
- **The slices in the wrong order,** or the $N^2$ values laid out column by column.

## Complexity

One projection of size $ND \times (2N + N^2)$ per token and sublayer, plus
$O(\texttt{iters} \cdot N^2)$ for the normalization and $O(N^2 D)$ for the merge. With
$N = 4$ the stream state is four times the hidden width per token, but attention and
the MLP still run at width $D$ on the collapsed vector, so the cost of a layer barely
changes.

## In the wild

Checked on 2026-11-21 against `models/deepseek_v4/modeling_deepseek_v4.py` of
`transformers` 5.23.0 and the `config.json` of `deepseek-ai/DeepSeek-V4-Pro-0813`
(`hc_mult` 4, `hc_sinkhorn_iters` 20, `hc_eps` 1e-6, `rms_norm_eps` 1e-6, 61 layers).

- `hc_weights` is `DeepseekV4HyperConnection.forward`: `input_norm` (an RMS norm
  without weight) on the flattened streams, `F.linear` with `fn`, the split into `N`,
  `N`, and `N * N` values, the three formulas, and the normalization loop exactly as
  in the statement. It returns `(post, comb, collapsed)`; this problem returns
  `collapsed` first.
- `hc_merge` is the expression in `DeepseekV4DecoderLayer.forward`:
  `post.unsqueeze(-1) * output.unsqueeze(-2) + matmul(comb.transpose(-1, -2), hidden_states)`.
  Its comment states the direction: `comb` "is consumed transposed".
- A layer has two hyper-connections, `attn_hc` and `ffn_hc`. The sublayer reads
  `input_layernorm(collapsed)` or `post_attention_layernorm(collapsed)`.
  `DeepseekV4Model` copies the embedding to all streams, and after the last layer
  `DeepseekV4HyperHead` collapses them with a `pre`-style weight (sigmoid plus `eps`)
  before the final norm.

The public tests compare with `DeepseekV4HyperConnection` at 1, 2, and 20 iterations
and with a `DeepseekV4DecoderLayer` whose sublayers are replaced by fixed outputs
(`fixtures/make.py`).

Simplifications, all of them: no batch axis; `iters` and `eps` are arguments instead
of configuration fields; the library computes the weights in float32 even when the
streams are in a lower precision and casts back, which this problem's float32 inputs
never exercise; the initializer's values (`base` zero, `scale` one) are replaced by
random ones in the tests so that each parameter shows.

The code attributes the method to "Xie et al., 2026" and to section 2.2 of the
DeepSeek-V4 paper (arXiv 2606.19348). Neither was read for this problem: everything
above is from the code, the configuration, and measurements on this machine.
