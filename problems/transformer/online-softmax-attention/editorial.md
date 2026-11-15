## The idea

For one query, attention is a ratio of two sums over the keys,

$$
\text{out} = \frac{\sum_j e^{s_j} v_j}{\sum_j e^{s_j}} ,
$$

and sums can be accumulated in any order, a block at a time. The only obstacle is
that $e^{s_j}$ is not representable: float32 overflows above $s = 88.7$ and
underflows below $-103$. `stable-softmax` fixes this by factoring out $e^{m}$ with
$m = \max_j s_j$, but here the maximum over *all* keys is not known until the last
block.

The online softmax (Milakov and Gimelshein, 2018) factors out the maximum *so far*
and repairs the sums when it changes. Keep three quantities per query:

$$
m = \max_{j \text{ seen}} s_j, \qquad
\ell = \sum_{j \text{ seen}} e^{s_j - m}, \qquad
o = \sum_{j \text{ seen}} e^{s_j - m}\, v_j .
$$

**Invariant.** $\ell\, e^{m}$ and $o\, e^{m}$ are the true sums over the keys seen so
far, so $o / \ell$ is attention over exactly those keys.

**Update.** A block with scores $s_j$, $j \in B$, raises the maximum to
$m' = \max(m, \max_{j \in B} s_j)$. Rewriting the old sums in the new unit multiplies
them by $e^{m - m'}$, and the block's terms are computed in the new unit directly:

$$
\ell' = \ell\, e^{m - m'} + \sum_{j \in B} e^{s_j - m'}, \qquad
o' = o\, e^{m - m'} + \sum_{j \in B} e^{s_j - m'}\, v_j .
$$

Substituting shows that the invariant holds for $m', \ell', o'$. By induction, after
the last block $o / \ell$ is attention over all keys, whatever the partition into
blocks: that is the property the tests check directly.

**Stability.** Every exponent is $\le 0$: $m - m' \le 0$ because the maximum never
decreases, and $s_j - m' \le 0$ by definition of $m'$. So nothing overflows. Once a
query has seen a visible key, the term of its current maximum is $e^0 = 1$ and
earlier terms are only ever scaled down, so $1 \le \ell \le N$: no division by zero.
Terms that underflow are below $10^{-45}$ of the largest and do not matter.

**The empty start.** Before a query's first visible key, $m = -\infty$, $\ell = 0$,
$o = 0$. If a block is fully masked for it, $m' = -\infty$ as well, and
$s_j - m' = -\infty - (-\infty)$ is NaN. Nothing is wrong mathematically (the block
contributes zero); the arithmetic needs one guard: subtract 0 instead of $-\infty$.
Then the block's terms are $e^{-\infty} = 0$, and at the first visible block the
rescale factor is $e^{-\infty - m'} = 0$, applied to sums that are 0.

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

- The state has one row per query; the maximum is per query, with the axis kept so it
  broadcasts against the scores.
- `acc` starts as the scalar 0 and takes the shape `(T, D_v)` from the first block,
  since $D_v$ is not known before.
- Nothing from a block outlives its iteration: `k`, `v`, `mask`, `scores`, and
  `weights` are rebound by the next one. The MLX reference adds
  `mx.eval(top, norm, acc)` at the end of each iteration. MLX builds a lazy graph,
  and without the evaluation the graph of the final result would reference every
  block's arrays until the very end, which is the memory use the algorithm exists to
  avoid.

## Common mistakes

- **No rescale when the maximum grows.** The sums from earlier blocks stay in the
  old unit and are overweighted by $e^{m' - m}$. Invisible when the first block
  holds the maximum, which is why a test with sorted or similar keys misses it.
- **Normalizing per block and averaging.** A softmax inside each block gives every
  block the same total weight. Attention over the union is not the mean of
  attentions over the parts; the parts must be weighted by their normalizers, which
  is what $\ell$ carries.
- **A block-local maximum.** Shifting each block by its own maximum is stable and
  wrong: the blocks end up in different units and are added as if they were not.
- **A plain running sum of `exp(score)`.** Correct on paper, `inf / inf` or `0 / 0`
  in float32 as soon as scores leave $[-103, 88]$.
- **One maximum for all queries.** The shift must be per query. With a shared
  maximum, a query whose scores are a few hundred below another's has every term
  underflow, and its output is `0 / 0`.
- **NaN from a fully masked block.** `exp(scores - new_top)` with both `-inf`. The
  NaN enters `norm` and `acc` and never leaves, since every later update multiplies
  it.
- **Ignoring the mask**, or applying it by multiplying weights with 0 after the
  exponential, which still lets a masked score set the maximum.
- **Two passes.** Computing the global maximum first and summing second needs the
  blocks twice. The iterator is exhausted after the first loop.
- **Collecting the blocks** (`list(kv_blocks)`, appending to lists, concatenating as
  you go) and running ordinary attention. The numbers are right and the memory is
  $O(N)$, the thing the problem rules out.

## Complexity

$O(T \cdot N \cdot (D + D_v))$ time, the same as ordinary attention. Memory is
$O(T \cdot D_v)$ for the state plus $O(T \cdot n)$ for the scores of one block,
independent of the number of keys.

## In the wild

The recurrence is from Milakov and Gimelshein, "Online normalizer calculation for
softmax" (arXiv 1805.02867, 2018), who used it to fuse softmax with a following
top-k in one pass over the logits. FlashAttention (Dao, Fu, Ermon, Rudra, and Ré,
arXiv 2205.14135, 2022) applies it to attention: its Algorithm 1 loops over blocks
of keys and values sized to fit in on-chip memory and keeps, per block of queries, a
running maximum $m$, a running normalizer $\ell$, and the partial output $O$, which
it rescales by $e^{m - m^{\text{new}}}$ at each step, exactly the update above (it
stores $O$ already divided by $\ell$ and renormalizes as it goes; FlashAttention-2
defers the division to the end, as the reference here does). The same update is what
lets attention be computed over a KV cache held in separate pages or on separate
devices: each part returns its partial $(m, \ell, o)$, and partial results combine
with the rule for two blocks.
