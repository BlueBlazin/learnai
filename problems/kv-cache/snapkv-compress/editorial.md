## The idea

H2O (the previous problem) decides what to evict as generation proceeds, from
attention accumulated over every step. SnapKV decides once, from a sample: the
queries of the last `window` prompt tokens. The paper's observation is that the
attention pattern of this window over the earlier prompt is stable through the
generation that follows, so the positions those queries attend to are the positions
later queries will want. The method is then three array operations per head.

*Vote.* Sum the window's rows over the prefix columns. The block
`attn[:, :, P:, :P]` is all that is read. Rows before the window are the wrong
sample (they describe how the document attends to itself, not how the question
attends to the document), and the window's own columns are not candidates: the
window is kept whole.

*Pool.* A raw top-k keeps isolated tokens. Attention often lands on one token of a
span (the first token of an entity, the delimiter before a value), and the tokens
next to it carry the content. Max-pooling with stride 1 gives every position the
best vote in its neighborhood, so a peak lifts its $r$ neighbors on each side to
its own level and they are selected together. The padding has to be neutral for a
maximum, which is $-\infty$, not something that could win; and the pooled array is
the prefix only, so nothing from the window leaks into the last prefix positions.

*Select.* Top $\text{budget} - \text{window}$ per head, then sort the positions so
that the compressed cache is still in sequence order, then append the window.

One consequence of pooling deserves a look. A peak and its neighbors have **equal**
pooled votes, so when the budget ends inside a plateau, the tie rule decides which
part of it survives. With "lowest position first" and a plateau `p - r .. p + r`
the left neighbors are taken before the peak itself; if only one slot is left, it
goes to `p - r`, not to `p`. The reference implementation uses `topk`, which makes
no promise about ties at all. A selection rule that wanted the peak first would
rank by the pair (pooled vote, raw vote); this problem keeps the method as
published and pins the tie rule down so that the result is defined.

## Reference solution (PyTorch)

```python
def snapkv_select(attn, window, budget, kernel):
    b, h, length, _ = attn.shape
    if length <= budget:
        return torch.arange(length).expand(b, h, length)
    prefix = length - window
    vote = attn[:, :, prefix:, :prefix].sum(dim=2)  # (B, H, prefix)
    half = kernel // 2
    padded = F.pad(vote, (half, half), value=float("-inf"))
    pooled = torch.stack([padded[..., o : o + prefix] for o in range(kernel)]).amax(dim=0)
    ranked = torch.argsort(pooled, dim=-1, descending=True, stable=True)
    chosen = torch.sort(ranked[..., : budget - window], dim=-1).values
    tail = torch.arange(prefix, length).expand(b, h, window)
    return torch.cat((chosen, tail), dim=-1)
```

The pooling is written as a maximum over `kernel` shifted views because that reads
the same in all three frameworks; `F.max_pool1d(vote, kernel, stride=1,
padding=kernel // 2)` is the same computation (it pads with $-\infty$ implicitly).
JAX and MLX use `pad(..., constant_values=-inf)` and `argsort(-pooled)`, which is
stable in both.

To compress a cache with the result, gather keys and values along the sequence axis
with the positions of each head, exactly as in `kv-eviction-heavy-hitters`.

## Common mistakes

- **No pooling.** The result is the raw top-k: scattered single tokens. The paper's
  ablation on a retrieval task finds this clearly worse than the pooled selection.
- **Votes summed over heads.** One selection for all heads is cheaper to store, and
  it is a different method. Heads attend to different things; per-head selection is
  the design.
- **The window's columns in the vote.** Recent tokens attend strongly to each other,
  so window positions win the top-k, use up the budget, and then appear a second
  time when the window is appended.
- **Pooling before cutting off the window.** The votes are right and the pooled
  values of the last $r$ prefix positions are not: they inherit the window's
  self-attention and are selected whatever the prefix says.
- **Every row votes.** The column sums of the whole matrix are the H2O score of a
  prefill. They favor early positions, which have been visible to more queries, and
  answer a different question.
- **Average pooling without thinking about the edges.** A mean with zero padding
  divides the first and last positions by `kernel` although fewer votes took part,
  and it lowers a sharp peak below a broad low plateau. (The paper reports that
  average and max pooling performed similarly in its experiments; this problem
  specifies max.)
- **`kernel` taken as a radius.** The neighborhood is `kernel` wide,
  $\lfloor \text{kernel}/2 \rfloor$ on each side.
- **Positions left in top-k order, or ties broken toward the end.** Both change
  which entry sits where in the compressed cache.

## Complexity

$O(B\,H\,\text{window}\,P)$ for the votes, $O(B\,H\,P\,\text{kernel})$ for the
pooling, and $O(B\,H\,P \log P)$ for the sort. It runs once per layer after the
prefill. In a real engine the block of attention it needs is not a by-product of
the prefill (fused attention kernels do not return weights), so the implementation
recomputes it from the last `window` queries and all keys, which is the
$\text{window} \times L$ product above and small next to the prefill itself.

## In the wild

The method is from "SnapKV: LLM Knows What You are Looking for Before Generation"
(arXiv 2404.14469). Its Listing 1 is this function: the vote is
`attn_weights[..., -window_size:, :-window_size].sum(dim=-2)`, the pooling is a 1-D
pool with `padding=kernel_size // 2` and `stride=1`, and the selection is
`topk(max_capacity_prompt - window_size)` per head, followed by a gather of keys and
values and a concatenation with the window. Three differences from this problem:
the listing gathers in the order `topk` returns, where this problem sorts the
positions; it leaves prompts shorter than the capacity untouched, which gives the
same result as the rule here; and it does not say which pooling, while the
experiments section reports max pooling with kernel sizes 5 and 7. The paper also
defines the number of kept positions through a compression rate,
$k = \lfloor p \cdot L_{\text{prefix}} \rfloor$; the listing's fixed capacity is
what this problem calls `budget`.
