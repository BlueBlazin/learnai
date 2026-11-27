## The idea

Heavy-hitter eviction combines two rules, each serving a distinct purpose.

**Keeping high-value older tokens.** A token's accumulated score is the sum of the attention it has received from every query since it entered the cache. For a prompt prefill, this is the column sum of the attention matrix. For a single-token decode step, it adds one new row of attention weights to the running totals. The H2O paper observed that these totals are strongly skewed. A small set of tokens collects most of the attention mass, and selecting them online from past steps closely matches an oracle that can see future attention.

**Protecting recent tokens.** Because scores accumulate over time, a token that arrived one step ago has had only one query to receive attention. Ranking purely by accumulated score would evict the newest tokens immediately. Reserving `recent` slots gives new tokens time to accumulate attention and keeps the immediate context that the next query attends to most strongly. The original paper splits the budget evenly between recent tokens and heavy hitters, while here `recent` is passed as a parameter.

**Keeping keys, values, and scores aligned.** After the first eviction, each head holds its own increasing subsequence of original tokens. The function therefore never tracks original token IDs. Instead, it operates purely on positions in the current cache, builds a single index tensor `kept` of shape `(B, H, N)`, and gathers `k`, `v`, and `scores` with that same tensor. Because `kept` is sorted in increasing order, sequence order is preserved by induction at every step, ensuring the last `recent` positions always correspond to the newest tokens.

When decoding one token at a time from a full cache ($S = \text{budget} + 1$), this rule evicts the single lowest-scoring entry outside the recent window per head, matching the greedy step in Algorithm 1 of the H2O paper. Using the same logic for any chunk size `T` also trims a long prompt down to `budget` in a single call.

## Reference solution (PyTorch)

```python
def h2o_step(cache, scores, attn_weights, budget, recent):
    k, v = cache
    b, h, t, s = attn_weights.shape
    received = attn_weights.sum(dim=2)
    if scores is None:
        scores = received
    else:
        scores = torch.cat((scores, torch.zeros(b, h, t)), dim=2) + received
    if s <= budget:
        kept = torch.arange(s).expand(b, h, s)
    else:
        old = s - recent
        ranked = torch.argsort(scores[:, :, :old], dim=-1, descending=True, stable=True)
        heavy = torch.sort(ranked[:, :, : budget - recent], dim=-1).values
        window = torch.arange(old, s).expand(b, h, recent)
        kept = torch.cat((heavy, window), dim=-1)
    index = kept[..., None].expand(-1, -1, -1, k.shape[-1])
    cache = (torch.gather(k, 2, index), torch.gather(v, 2, index))
    return cache, torch.gather(scores, 2, kept), kept
```

A stable descending sort enforces the tie-breaking rule because equal scores keep their original position order, placing the older position first. `torch.topk` is shorter but does not guarantee tie order. In JAX and MLX, `argsort` is stable, so you can rank with `argsort(-scores)` and gather with `take_along_axis`.

Neither `recent = budget` nor `recent = 0` needs a special case because `ranked[:, :, :0]` and `arange(s, s)` are empty tensors that concatenate cleanly.

## Common mistakes

- **Ranking by the latest attention weights.** A single query row is noisy. A token that the model consults every few steps has a high accumulated score even when its latest weight is zero.
- **Not gathering `scores`.** Returning an un-gathered slice of `scores` leaves scores attached to slot indices rather than tokens. After the first eviction, every score after the evicted slot points to the wrong entry.
- **Scrambling sequence order.** `topk` and `argsort` return positions sorted by score rather than position. Gathering in score order breaks causal order and invalidates "the last `recent` positions" on the next step.
- **Omitting the recent window.** Taking pure top-`budget` scores evicts newly appended tokens before they can accumulate attention.
- **Sharing one ranking across all heads.** Summing scores across heads simplifies storage, but heads specialize in different tokens and one head's heavy hitter is irrelevant to another.
- **Breaking ties toward newer tokens.** An unstable or reversed sort can keep the higher position index on a tie. Exact ties are common early on when multiple entries have score `0`.
- **Using only the last query row of a chunk.** During prefill, `attn_weights[:, :, -1]` ignores the first `T - 1` queries. The accumulated score must sum over all `T` rows.

## Complexity

Each step takes $O(B\,H\,T\,S)$ to accumulate scores, $O(B\,H\,S \log S)$ to sort the older positions, and $O(B\,H\,N\,D)$ to gather the kept cache. Production implementations avoid sorting and copying on single-token decode steps by finding the single minimum-scoring slot outside the recent window and overwriting it in place, using a preallocated buffer with a circular queue for the recent window.

Memory usage is fixed at `budget` entries per head and layer regardless of sequence length, plus one float per entry for its accumulated score.

## In the wild

This method comes from "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" (arXiv 2306.14048, NeurIPS 2023). The paper defines heavy hitters by accumulated attention, balances recent tokens and heavy hitters (splitting the budget evenly in its experiments), and evicts independently per head so that one heavy hitter is evicted per head at each decode step.

Because eviction is permanent, a discarded token cannot be recovered if the model later needs it, which is why the budget and window size are tuned per task. Related KV eviction policies select survivors differently, such as a sliding window with attention sinks (see `sliding-window-cache`) or one-shot prompt compression from the last prompt tokens (see `snapkv-compress`).
