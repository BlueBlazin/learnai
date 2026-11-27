Break `serve` into three separate questions, since each relies on different state. First, which blocks can be reused at all? (Only the leading contiguous run of stored blocks.) Second, how many of them are worth reusing? (Compare total times, remembering that a tier's base latency is paid once per tier, not once per block.) Third, which blocks leave the GPU and where do they go? (Track recency, block depth, and free slots in lower tiers.) What metadata does your index need per stored block to answer all three?

---

Store three fields for every cached block: its current tier, its block index (depth), and a `stamp` recording the number of the last `serve` call that used it. Then implement each piece directly:

- **Cut selection:** For each `c` from `0` to `p`, count the host and disk blocks among the first `c` blocks, compute `t(c)`, and update the best cut with `<=` so larger cuts win ties.
- **Victim selection:** Pick the unpinned block in the target tier that minimizes `(stamp, -depth)`.
- **Placement:** Write a recursive helper that frees a slot in the target tier by placing that tier's victim one tier lower before demoting the incoming block.

---

```
serve(tokens):
    keys = chained keys of the full blocks;  pinned = set(keys)
    p = number of leading keys in the index
    c = argmin over 0..p of t(c), largest c on ties
    for key in keys[c:]: if stored: store.drop(key); forget it
    need = m - (reused blocks on the GPU)
    while free("gpu") < need: place(victim("gpu"), "host")
    for tier in ("host", "disk"):
        run = reused keys in that tier
        if run: await store.load(run); mark them "gpu"
    for k in c..m-1: await store.compute(keys[k], keys[k-1] or None, block k's tokens)
    stamp all m blocks with this call's number

place(key, tier):
    if tier is None: store.drop(key); forget it; return
    if free(tier) == 0:
        v = victim(tier)
        if v is None: return place(key, below(tier))
        place(v, below(tier))
    store.demote(key, tier); record the new tier
```

Update your index immediately with every store call so `free` and `victim` always reflect the current store state.
