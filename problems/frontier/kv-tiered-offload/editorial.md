## The idea

A single-tier prefix cache only decides which block to evict. A tiered store adds two more decisions, and each one maps to a small mechanism.

**Demoting blocks under pressure.** When a block must leave the GPU, it still has value, so the policy moves it down a tier rather than deleting it. If the target tier is full, that tier's least valuable block is moved down first. The `place` routine expresses this as a recursion: free a slot in the tier below, then move the block in. Only pushing a block below `"disk"` deletes it.

Value is ordered by recency, with one tie-breaker specific to prefix chains. Every block in a prompt is used by the same `serve` call, so all blocks of that prompt share the same timestamp. Breaking ties in favor of the **deepest** block reflects dependency order. Because a block is usable only if every earlier block is present, the tail of a chain is less valuable than its head. Demoting a parent before its child would leave the child stranded in a hotter tier. Deepest-first eviction maintains a key invariant: along any prefix chain, tiers only get colder with depth. As a result, the reused prefix of a prompt always consists of GPU blocks, followed by host blocks, followed by disk blocks. One batched load per tier brings them back, and every block's parent is already on the GPU when it arrives.

**Choosing the reusable prefix.** An engine can only skip a contiguous *prefix* of prefill because attention at each position depends on all earlier tokens. For a prompt of $m$ full blocks, the policy picks a single cut $c$, reusing blocks $0, \ldots, c - 1$ and computing blocks $c, \ldots, m - 1$. The search stops at the first missing block $p$, since nothing behind a gap can be reused even if later blocks still sit in a colder tier.

**Deciding whether to load or recompute.** Reusing a block on the GPU is free. Reusing blocks from host or disk incurs a fixed transfer setup cost plus a per-block cost, while recomputing costs a fixed time per block. Whether loading wins depends on how many blocks come from that tier. With the example's disk costs, loading 2 blocks takes $0.75$ s versus $0.5$ s to compute, whereas loading 6 blocks takes $1.25$ s versus $1.5$ s to compute. You also cannot decide one tier at a time, because a short host run that loses on its own may unlock a long disk run behind it. Evaluating the total time $t(c)$ for every cut $c \in \{0, \ldots, p\}$ and taking the minimum handles all combinations directly.

**Pinning active blocks.** While a request is being served, none of its blocks may be evicted from any tier. Its GPU blocks are about to be reused, and its lower-tier blocks are about to be loaded. Pinning introduces the third branch of `place`: when a target tier is full of pinned blocks, the block being demoted skips that tier and is placed in the next tier down.

## Reference solution (Python)

```python
BELOW = {"gpu": "host", "host": "disk", "disk": None}


class Block:
    def __init__(self, tier, depth):
        self.tier, self.depth, self.stamp = tier, depth, 0


class TieredKV:
    def __init__(self, store, block_size):
        self.store, self.block_size = store, block_size
        self.blocks = {}  # key -> Block
        self.serves = 0

    def _keys(self, tokens):
        size, keys, parent = self.block_size, [], b""
        for start in range(0, len(tokens) - size + 1, size):
            block = struct.pack(f"<{size}q", *tokens[start : start + size])
            parent = hashlib.sha256(parent + block).digest()
            keys.append(parent)
        return keys

    def _free(self, tier):
        used = sum(1 for block in self.blocks.values() if block.tier == tier)
        return self.store.capacity[tier] - used

    def _victim(self, tier, pinned):
        candidates = [
            (block.stamp, -block.depth, key)
            for key, block in self.blocks.items()
            if block.tier == tier and key not in pinned
        ]
        return min(candidates)[2] if candidates else None

    def _place(self, key, tier, pinned):
        if tier is None:
            self.store.drop(key)
            del self.blocks[key]
            return
        if self._free(tier) == 0:
            victim = self._victim(tier, pinned)
            if victim is None:
                self._place(key, BELOW[tier], pinned)
                return
            self._place(victim, BELOW[tier], pinned)
        self.store.demote(key, tier)
        self.blocks[key].tier = tier

    async def serve(self, tokens):
        store, costs, size = self.store, self.store.costs, self.block_size
        keys = self._keys(tokens)
        total = len(keys)
        self.serves += 1
        pinned = set(keys)

        stored = 0
        while stored < total and keys[stored] in self.blocks:
            stored += 1
        tiers = [self.blocks[key].tier for key in keys[:stored]]

        cut, best = 0, None
        for candidate in range(stored + 1):
            host = tiers[:candidate].count("host")
            disk = tiers[:candidate].count("disk")
            seconds = costs["compute_s"] * (total - candidate)
            if host:
                seconds += costs["host_base_s"] + costs["host_block_s"] * host
            if disk:
                seconds += costs["disk_base_s"] + costs["disk_block_s"] * disk
            if best is None or seconds <= best:
                cut, best = candidate, seconds

        for key in keys[cut:]:
            if key in self.blocks:
                store.drop(key)
                del self.blocks[key]

        result = {tier: tiers[:cut].count(tier) for tier in ("gpu", "host", "disk")}
        result["computed"] = total - cut
        need = total - result["gpu"]
        while self._free("gpu") < need:
            self._place(self._victim("gpu", pinned), "host", pinned)

        for tier in ("host", "disk"):
            run = [key for key in keys[:cut] if self.blocks[key].tier == tier]
            if run:
                await store.load(run)
                for key in run:
                    self.blocks[key].tier = "gpu"

        for index in range(cut, total):
            parent = keys[index - 1] if index else None
            await store.compute(keys[index], parent, tokens[index * size : (index + 1) * size])
            self.blocks[keys[index]] = Block("gpu", index)

        for key in keys:
            self.blocks[key].stamp = self.serves
        return result

    def discard(self, tokens):
        keys = self._keys(tokens)
        if not keys or keys[-1] not in self.blocks:
            return False
        self.store.drop(keys[-1])
        del self.blocks[keys[-1]]
        return True
```

The index is updated right alongside every store operation so `_free` and `_victim` always reflect the exact state of the store. Linear scans over `self.blocks` are fast enough at these capacities. A production cache maintains a recency structure per tier.

## Common mistakes

- **Evicting instead of demoting.** Dropping GPU victims directly reduces the policy to a single-tier cache, leaving host and disk unused and forcing returning prompts to prefill from scratch.
- **Always loading stored blocks.** Treating every stored block as a hit ignores transfer overhead and slows down short prompts when recomputing is faster than loading from disk.
- **Deciding tier by tier.** Checking whether the host run pays for itself independently of the disk run misses cases where loading a short host run unlocks a profitable disk run behind it.
- **Breaking ties toward recomputation.** Using `<` instead of `<=` picks the smaller cut on a tie, violating the rule that ties favor reuse.
- **Loading past a missing block.** Reusing stored blocks after a gap tries to load a block whose ancestor is missing from the GPU, which the store rejects.
- **Hashing a block without its parent.** Identical tokens after different prefixes produce different KV states. Ignoring the prefix serves one conversation another's cache.
- **Demoting shallowest first.** Moving the head of a chain before its tail breaks tier ordering along the chain, placing the earliest blocks on the slowest tier.
- **Forgetting to pin active blocks.** Without pinning, making room on the GPU can demote blocks that the current request is about to load or reuse on the GPU.
- **Skipping the cascade.** Sending a GPU victim straight to disk when `"host"` is full keeps stale blocks in fast host memory while pushing fresher blocks to slow disk.
- **Not updating recency on GPU hits.** Refreshing timestamps only for loaded and computed blocks turns GPU eviction into FIFO, demoting a frequently reused system prompt first.
- **Loading one block per call.** Issuing one `store.load` per block pays the tier's fixed base latency repeatedly instead of once per tier.

## Complexity

Each `serve` call computes $O(m)$ block keys, evaluates $O(m)$ candidate cuts ($O(m^2)$ as written because counts are recomputed per cut), and scans the block index for each victim. The chosen cut minimizes simulated store time. Total bytes moved equals `store.block_bytes` times the number of loads and demotions. For example, a block that moves from GPU to host to disk and back to GPU is counted three times.

## In the wild

What is established in production systems:

- **LMCache** (architecture documentation and "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference", arXiv 2510.09665) implements this storage hierarchy for engines like vLLM. GPU memory holds the active working set, pinned CPU memory acts as a hot cache for fast transfers, local storage provides a larger tier, and remote storage offers slower persistent capacity. Overflow offloads from GPU to CPU and then to disk or remote storage under LRU eviction, and chunks (256 tokens by default) are indexed by hash and loaded back on demand.
- **NVIDIA Dynamo's KV block manager (KVBM)** defines tiers G1 (GPU device pool), G2 (pinned host CPU pool), G3 (local NVMe disk pool), and G4 (remote storage). It distinguishes *offload* (device to host, host to disk) from *onboard* (host to device, or disk directly to device without staging through host memory). The direct disk-to-GPU load in this problem follows that model.
- **vLLM**'s automatic prefix caching uses parent-chained block hashes for block identity, as in `block-hash-prefix-cache`.

What this problem simplifies or chooses:

- **Exclusive tiers.** Real stores often retain a lower-tier copy when loading a block to the GPU, making subsequent demotion free. Exclusive tiers keep `bytes_moved` unambiguous and allow a host tier full of pinned blocks to occur.
- **Zero-latency demotions and serial execution.** Production systems offload asynchronously off the critical path. Here demotions charge bytes rather than time, and loads and computations execute serially for one request at a time.
- **Linear cost model and cut rule.** Modeling transfer time as `base + per_block * n` against a fixed `compute_s` per block abstracts bandwidth and prefill speed. Minimizing $t(c)$ is this problem's formalization of the load-versus-recompute tradeoff rather than a literal description of LMCache or Dynamo.
- **Deepest-first tie-breaking.** Production systems document LRU eviction across tiers, whereas breaking same-call ties deepest-first is chosen here to preserve monotonic tier ordering along chains.
- **`discard` for out-of-order removal.** `discard` models expiry, manual invalidation, or storage loss, which is the only way a stored block can sit behind a missing ancestor under this policy.
