## The idea

For a single query, attention is the ratio of two sums over all visible keys:

$$
\text{out} = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}}.
$$

Sums can be accumulated one block at a time in any order. The obstacle is floating-point range. In `float32`, $e^{s_j}$ overflows above $s_j \approx 88.7$ and underflows below $s_j \approx -103$. `stable-softmax` avoids this by factoring out $e^m$ where $m = \max_j s_j$, but in a streaming pass the global maximum is unknown until the last block.

Online softmax (Milakov and Gimelshein, 2018) factors out the maximum *seen so far* and rescales the accumulated sums whenever that maximum grows. For each query, track three quantities:

$$
m = \max_{j \text{ seen}} s_j, \qquad
\ell = \sum_{j \text{ seen}} e^{s_j - m}, \qquad
o = \sum_{j \text{ seen}} e^{s_j - m}\, v_j.
$$

**Invariant.** At every step, $\ell\, e^{m}$ and $o\, e^{m}$ equal the unshifted denominator and numerator over all keys seen so far. Their ratio $o / \ell$ is therefore exact attention over those keys.

**Update.** When a new block $B$ arrives with scores $s_j$ ($j \in B$), the new maximum is $m' = \max(m, \max_{j \in B} s_j)$. Multiplying the old sums by $e^{m - m'}$ converts them to the new scale $e^{m'}$, and the new block's terms are computed directly in that scale:

$$
\ell' = \ell\, e^{m - m'} + \sum_{j \in B} e^{s_j - m'}, \qquad
o' = o\, e^{m - m'} + \sum_{j \in B} e^{s_j - m'}\, v_j.
$$

Substituting the definitions confirms that the invariant holds for $(m', \ell', o')$. By induction, $o / \ell$ after the last block equals attention over all keys regardless of how they are partitioned into blocks.

**Stability.** Every exponent is at most 0. Because $m$ never decreases, $m - m' \le 0$. By definition of $m'$, $s_j - m' \le 0$. Nothing overflows. Once a query has seen at least one visible key, the term achieving the current maximum contributes $e^0 = 1$, and earlier positive terms are only scaled down. Thus $1 \le \ell \le N$, preventing division by zero. Terms that underflow are smaller than $10^{-45}$ times the largest term and do not affect `float32` results.

**The empty start.** Before a query sees its first visible key, $m = -\infty$, $\ell = 0$, and $o = 0$. If a block is fully masked for that query, $m'$ remains $-\infty$, so $s_j - m'$ evaluates to $-\infty - (-\infty) = \text{NaN}$. Guard against this by replacing $-\infty$ with $0$ in the shift. Masked scores then give $e^{-\infty - 0} = 0$, leaving $\ell$ and $o$ at 0. When the first visible block arrives with finite $m'$, the rescale factor is $e^{-\infty - m'} = 0$, which safely multiplies the zero sums.

## Reference solution (PyTorch)

```python
def blockwise_attention(q, kv_blocks):
    n_queries, d = q.shape
    top = torch.full((n_queries, 1), float("-inf"))
    norm = torch.zeros(n_queries, 1)
    acc = 0.0

    for k, v, mask in kv_blocks:
        scores = torch.where(mask, (q @ k.T) * d**-0.5, float("-inf"))
        new_top = torch.maximum(top, scores.amax(dim=-1, keepdim=True))
        shift = torch.where(torch.isinf(new_top), 0.0, new_top)
        rescale = torch.exp(top - shift)
        weights = torch.exp(scores - shift)
        norm = norm * rescale + weights.sum(dim=-1, keepdim=True)
        acc = acc * rescale + weights @ v
        top = new_top

    return acc / norm
```

- The running state keeps one row per query with a trailing size-1 axis so it broadcasts against the block scores.
- `acc` starts as scalar `0.0` and takes shape `(T, D_v)` on the first block, since $D_v$ is not known beforehand.
- No block tensor outlives its loop iteration because `k`, `v`, `mask`, `scores`, and `weights` are rebound on the next step. The MLX reference calls `mx.eval(top, norm, acc)` at the end of each iteration. Because MLX builds a lazy computation graph, skipping `mx.eval` would keep every block's arrays alive in the graph until the final return.

## Common mistakes

- **Skipping the rescale when the maximum grows.** Earlier sums remain in the old scale and are overweighted by $e^{m' - m}$. This bug is invisible whenever the first block already contains the global maximum.
- **Normalizing per block and averaging.** Taking a softmax inside each block gives every block equal total weight. Attention over the union must weight each block by its unnormalized sum $\ell$, not average block outputs.
- **Using a block-local maximum.** Shifting each block by its own maximum avoids overflow, but adds sums that are in incompatible scales.
- **Accumulating unshifted `exp(score)`.** This works on paper, but produces `inf / inf` or `0 / 0` in `float32` as soon as scores leave $[-103, 88]$.
- **Sharing one maximum across all queries.** The shift must be per query. With a shared maximum, a query whose scores sit hundreds below another query's scores underflows completely and returns `0 / 0`.
- **NaN from a fully masked block.** Computing `exp(scores - new_top)` when both are `-inf` introduces `NaN` into `norm` and `acc`. Every subsequent update propagates that `NaN`.
- **Ignoring the mask or zeroing after `exp`.** Multiplying weights by 0 after exponentiation still lets masked scores affect `new_top` and shift valid scores into underflow.
- **Making two passes.** Finding the global maximum in one loop and summing in a second fails because `kv_blocks` is exhausted after the first pass.
- **Collecting all blocks.** Calling `list(kv_blocks)` or concatenating blocks along the key axis produces the right values, but uses $O(N)$ memory and violates `peak_blocks_held`.

## Complexity

Time is $O(T \cdot N \cdot (D + D_v))$, identical to standard attention. Memory is $O(T \cdot D_v)$ for the running state plus $O(T \cdot n)$ for one block's scores, independent of the total key count $N$.

## In the wild

The online normalizer recurrence comes from Milakov and Gimelshein, "Online normalizer calculation for softmax" (arXiv 1805.02867, 2018), who used it to fuse softmax and top-k into a single pass over logits. FlashAttention (Dao, Fu, Ermon, Rudra, and Ré, arXiv 2205.14135, 2022) applies it to attention. Its Algorithm 1 tiles keys and values into on-chip memory and maintains a running maximum $m$, normalizer $\ell$, and partial output $O$ per query block, rescaling by $e^{m - m^{\text{new}}}$ at each step. FlashAttention-1 stores $O$ already divided by $\ell$ and renormalizes on every step, while FlashAttention-2 defers the division to the end as in the reference above. The same associative update allows attention over paged or distributed KV caches. Each partition returns its partial $(m, \ell, o)$, and partitions merge using the two-block rule.
