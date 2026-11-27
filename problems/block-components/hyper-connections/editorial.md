## The idea

A standard residual stream carries one vector per token that every sublayer reads from and adds back to. Hyper-connections widen that stream to $N$ vectors and make reading, writing, and carrying input-dependent:

- **Read.** The sublayer receives a weighted combination of the streams, $\sum_n \texttt{pre}_n X_n$.
- **Write.** The sublayer output $y$ is added to stream $n$ with weight $\texttt{post}_n$.
- **Carry.** Instead of copying old streams unchanged as in $x + f(x)$, the layer mixes them via $X' = \texttt{comb}^\top X$.

Setting $N = 1$ and $\texttt{pre} = \texttt{post} = \texttt{comb} = 1$ recovers the standard residual connection. All three sets of weights come from a single linear projection of the normalized, flattened streams, letting each token route information dynamically.

The carry term requires extra care because it runs at every sublayer. Across a deep network, the product of all mixing matrices sits directly between the embedding and the output. An unconstrained matrix would let that product explode or vanish with depth. To prevent this, the manifold-constrained variant pushes `comb` toward a doubly stochastic matrix (non-negative entries with every row and column summing to 1) via Sinkhorn-Knopp normalization, which alternately rescales columns and rows.

## What the iteration guarantees, and what it only approaches

The final Sinkhorn step is always a column normalization, and `hc_merge` multiplies by `comb` transposed so that column $n$ holds the weights for new stream $n$. Together, these guarantee an exact invariant for any `iters >= 1`. Every new stream is a positive weighted average of the old streams whose weights sum to $s / (s + \texttt{eps})$, where $s$ is the column sum before the last step. As a result, the carry term cannot increase the largest stream norm.

Row sums behave differently. A row sum measures how much of old stream $j$ survives across all new streams, and it only approaches 1 in the limit. How close 20 iterations get depends on how peaked the initial softmax is. Measured on 200,000 random and 200,000 two-valued $4 \times 4$ matrices of softmax inputs spanning a given range (float64, `eps = 1e-6`), the largest row-sum error after 20 iterations is:

| span of the softmax inputs | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| largest row-sum error | 1e-6 | 9e-4 | 6e-3 | 2e-2 | 3e-2 |

Double stochasticity is therefore exact for columns and only approximate for rows. For sharp matrices, 20 iterations are still far from converged. This has two practical consequences:

- Changes to the starting point—such as normalizing rows first, taking the softmax over columns, or running an extra round—converge to the same matrix up to `eps` terms, because all of them are diagonal rescalings of the same positive matrix and the doubly stochastic limit is unique. Even so, they are wrong here. At 1 or 2 iterations, and at 20 iterations for sharp inputs, their outputs differ in the first or second digit. The tests use 1, 2, and 3 iterations to catch these order bugs.
- Ending on a row step instead of a column step moves the exact sum-to-1 guarantee to the wrong axis. After the transpose in `hc_merge`, the new streams are no longer convex combinations of the old ones.

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

- **Using `comb @ streams` without transposing.** Without the transpose, new stream $n$ uses row $n$ of `comb`, whose sum is only approximately 1. For symmetric matrices nothing changes, and every $2 \times 2$ doubly stochastic matrix is symmetric, so a converged two-stream test cannot catch this bug.
- **Wrong Sinkhorn order or count.** Skipping Sinkhorn, starting with rows, running one round too many, or taking the softmax over columns all fail at low iteration counts and on sharp inputs.
- **Omitting the factor of 2 on `post`.** Without it, the sigmoid midpoint is 0.5 instead of 1.0, so a freshly initialized model writes only half of each sublayer output.
- **Averaging the streams instead of weighting by `pre`.** The collapsed vector is a `pre`-weighted sum, not a uniform mean.
- **Normalizing each stream separately.** Taking an RMS norm over each stream's $D$ channels equalizes all stream norms before the projection. Normalizing the flattened $ND$ vector preserves the relative magnitudes of the streams.
- **Misusing `eps`.** `eps` is added to `pre` (so no stream turns off completely), added to the softmax output (so no entry of `comb` is zero, which Sinkhorn could never recover), and added to each Sinkhorn divisor. It is not added to `post`, and it is not the RMS norm epsilon (`1e-6`). Both equal `1e-6` in the shipped config, so the mix-up hides unless tested with a larger `eps` like `0.01`.
- **Computing `(raw + base) * scale`.** `scale` multiplies only the projected value `raw`, before adding `base`.
- **Wrong slice order or reshape layout.** The slices are `pre`, `post`, then `comb`, and the $N^2$ entries of `comb` are laid out row by row, not column by column.

## Complexity

Each token and sublayer performs one linear projection of size $ND \times (2N + N^2)$, plus $O(\texttt{iters} \cdot N^2)$ work for Sinkhorn normalization and $O(N^2 D)$ for the merge. With $N = 4$, the residual state is four times larger per token, but attention and the MLP still run at width $D$ on the collapsed vector, so total layer compute barely changes.

## In the wild

Verified on 2026-11-21 against `models/deepseek_v4/modeling_deepseek_v4.py` in `transformers` 5.23.0 and `config.json` of `deepseek-ai/DeepSeek-V4-Pro-0813` (`hc_mult = 4`, `hc_sinkhorn_iters = 20`, `hc_eps = 1e-6`, `rms_norm_eps = 1e-6`, 61 layers).

- `hc_weights` matches `DeepseekV4HyperConnection.forward`: `input_norm` (a weightless RMS norm) on the flattened streams, `F.linear` with `fn`, splitting into `N`, `N`, and `N * N` values, the three formulas, and the exact Sinkhorn loop. The library returns `(post, comb, collapsed)`, whereas this problem returns `collapsed` first.
- `hc_merge` matches `DeepseekV4DecoderLayer.forward`, which computes `post.unsqueeze(-1) * output.unsqueeze(-2) + matmul(comb.transpose(-1, -2), hidden_states)` and notes in a comment that `comb` "is consumed transposed".
- Each decoder layer has two hyper-connections (`attn_hc` and `ffn_hc`), and each sublayer reads `input_layernorm(collapsed)` or `post_attention_layernorm(collapsed)`. `DeepseekV4Model` copies the token embedding to all streams, and `DeepseekV4HyperHead` collapses them after the last layer with a `pre`-style weight (sigmoid plus `eps`) before the final norm.

The public tests compare directly against `DeepseekV4HyperConnection` at 1, 2, and 20 iterations, and against a `DeepseekV4DecoderLayer` whose sublayers are replaced by fixed outputs (`fixtures/make.py`).

Simplifications from the shipped implementation: the batch axis is omitted, and `iters` and `eps` are passed as arguments rather than read from config. The library also upcasts to float32 when inputs are lower precision, which this problem's float32 inputs do not exercise. Finally, the tests replace the default initialization (`base = 0`, `scale = 1`) with random values so every parameter is exercised.

The code attributes the method to Xie et al. (2026) and section 2.2 of the DeepSeek-V4 paper (arXiv 2606.19348). Everything above was verified directly from the code, configuration, and local measurements.
