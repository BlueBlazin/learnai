## The idea

H2O (from the previous problem) evicts entries step by step during generation using attention accumulated over every step. SnapKV decides once, right after prefill, using the queries of the last `window` prompt tokens. The attention pattern of this observation window over the earlier prompt stays stable through generation, so the positions those queries attend to are the ones later queries will need. The selection reduces to three array operations per head.

**Vote.** Sum the observation window's rows over the prefix columns, where $P = L - \text{window}$. Only the slice `attn[:, :, P:, :P]` is read. Earlier query rows describe how the document attends to itself rather than how the question attends to the document, and the window's own columns are not candidates because the window is always kept intact.

**Pool.** A raw top-k selects isolated tokens. Attention often lands on a single anchor token of a span, such as the first token of an entity or a delimiter before a value, while adjacent tokens hold the rest of the content. Max-pooling with stride 1 assigns each position the highest vote in its neighborhood of radius $r = \lfloor \text{kernel} / 2 \rfloor$, lifting a peak's neighbors to the same score so they are selected together. Padding at the prefix boundaries must be $-\infty$ so that out-of-bounds slots never win, and pooling must run on the prefix alone so window weights cannot leak into the last prefix positions.

**Select.** Take the top $\text{budget} - \text{window}$ prefix positions per head, sort those indices into increasing sequence order, and append the window positions $P, \dots, L - 1$.

One subtlety of pooling is how ties behave. A peak and its $2r$ neighbors share the exact same pooled vote, so when the budget ends inside a plateau, the tie-breaking rule decides which part survives. Under "lowest position first," a plateau `p - r .. p + r` picks its left neighbors before the peak itself. If only one slot remains, it goes to `p - r` rather than `p`. The original paper's code uses `topk`, which leaves tie order unspecified. A rule that prefers the peak first would rank by `(pooled_vote, raw_vote)`. This problem follows the published algorithm and fixes the tie rule so the output is deterministic.

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

Writing the pooling as a maximum over `kernel` shifted views works the same way across all three frameworks. In PyTorch, `F.max_pool1d(vote, kernel, stride=1, padding=kernel // 2)` computes the same result because it pads with $-\infty$ implicitly. JAX and MLX use `pad(..., constant_values=-inf)` and `argsort(-pooled)`, which is stable in both frameworks.

To compress an actual cache with the returned indices, gather keys and values along the sequence axis per head, just as in `kv-eviction-heavy-hitters`.

## Common mistakes

- **Skipping pooling.** Taking raw top-k selects scattered single tokens. The SnapKV paper's retrieval ablation shows this performs clearly worse than pooled selection.
- **Summing votes across heads.** Sharing one set of positions across all heads is simpler to store, but different heads attend to different parts of the prompt. Per-head selection is core to the method.
- **Including the window's columns in the vote.** Recent tokens attend heavily to each other, so window positions win the top-k, consume budget slots, and then get duplicated when the window is appended.
- **Pooling before slicing off the window.** Even if the votes are summed properly, pooling across the boundary lets the last $r$ prefix positions inherit the window's high self-attention and get selected regardless of their own votes.
- **Letting every query row vote.** Summing all rows of `attn` computes the H2O prefill score instead. That biases toward early tokens (which are visible to more queries) rather than what the end of the prompt attends to.
- **Using average pooling with zero padding.** Dividing by `kernel` at the edges penalizes boundary positions where fewer votes exist, and averaging lowers a sharp peak below a broad low plateau. (The paper notes that average and max pooling performed similarly in its experiments, while this problem specifies max pooling.)
- **Treating `kernel` as a radius.** The pooling window has total width `kernel`, which is $\lfloor \text{kernel} / 2 \rfloor$ positions on each side.
- **Leaving positions in top-k order or breaking ties toward the end.** Both change the order or membership of entries in the compressed cache.

## Complexity

Computing votes takes $O(B\,H\,\text{window}\,P)$ time, pooling takes $O(B\,H\,P\,\text{kernel})$, and sorting takes $O(B\,H\,P \log P)$. This runs once per layer after prefill. In a production engine, fused attention kernels do not materialize the full attention matrix, so SnapKV recomputes just the $\text{window} \times L$ attention scores from the last `window` queries and all keys. That extra work is small compared to the full prefill.

## In the wild

This method comes from "SnapKV: LLM Knows What You are Looking for Before Generation" (arXiv 2404.14469). Listing 1 in the paper matches this function closely. It votes with `attn_weights[..., -window_size:, :-window_size].sum(dim=-2)`, applies 1-D pooling with `padding=kernel_size // 2` and `stride=1`, selects `topk(max_capacity_prompt - window_size)` per head, gathers keys and values, and concatenates the window. There are three minor differences from this problem. Listing 1 gathers in `topk` order rather than sorting the selected indices. It leaves prompts shorter than capacity untouched, which matches the `L <= budget` rule here. Finally, Listing 1 omits the pooling type, while the experiments section uses max pooling with kernel sizes 5 and 7. The paper also parameterizes kept prefix length via a compression rate $k = \lfloor p \cdot L_{\text{prefix}} \rfloor$, whereas Listing 1 uses a fixed capacity called `budget` here.
