Separate the three questions `serve` answers, because each has its own data. Which
blocks can be reused at all? (Only a leading run of stored blocks.) How many of
them are worth reusing? (A comparison of times, where a transfer's base cost is
paid once per tier, not per block.) What has to leave the GPU to make room, and
where does it go? (Recency, depth, and the free slots of the tiers below.) What
does your index need to remember about a block to answer all three?

---

Keep one record per stored block: its tier, its index in the chain (depth), and a
stamp, the number of the last `serve` call that used it. Then:

- the cut: for every `c` from 0 to `p`, count the host and disk blocks among the
  first `c`, compute `t(c)`, and keep the best with `<=` so that later (larger)
  cuts win ties;
- the victim of a tier: the minimum of `(stamp, -depth)` over its unpinned blocks;
- place: a short recursive function that first makes room in the target tier by
  placing that tier's victim one tier further down.

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

Update your index at the same moment you call the store, so that `free` and
`victim` always describe the store as it is.
