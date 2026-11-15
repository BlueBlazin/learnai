## The idea

A prefix cache on one tier has one decision, what to evict. With tiers below the
GPU there are three, and the solution is one small mechanism for each.

**Where blocks go under pressure.** A block that must leave the GPU is still worth
something, so it is moved down rather than deleted, and if the tier below is full,
that tier's least valuable block is moved down first. "Place" is this recursion:
make room below, then move. The disk is the only tier that deletes.

Which block is least valuable is recency, with one refinement that matters for
prefix blocks. All blocks of a prompt are used by the same call, so they tie, and
the tie must go to the **deepest** block. A block is only usable when every block
before it is available, so the tail of a chain is worth less than its head, and a
parent that was demoted before its child would leave the child stranded in a hot
tier. Deepest-first gives an invariant: along any chain the tiers only get colder
with depth. Everything else leans on it. The reused part of a prompt is a run of
GPU blocks, then host blocks, then disk blocks; one transfer per tier brings them
back; and each transfer finds the parents of its blocks already on the GPU.

**What a request reuses.** An engine can skip the *prefix* of a prefill and nothing
else: the keys and values of a position are only useful when all earlier positions
are present. So the choice is a single cut $c$: reuse blocks before it, compute
from it on. The search stops at the first block that is stored nowhere, because
nothing behind it can be reached, whatever is still lying in a cold tier.

**Load or recompute.** Reusing a block on the GPU is free. Reusing a block from a
lower tier costs a share of a transfer, and a transfer has a fixed part (a request
to another process or device, a file to open) and a part per block. Computing costs
a fixed time per block. So the answer depends on how many blocks a tier would
deliver: with the example's disk costs, two blocks are not worth fetching
($0.75$ s against $0.5$ s) and six are ($1.25$ s against $1.5$ s). The decision
cannot be made run by run, either: a host run that does not pay for itself may
still be worth loading because it gives access to a long disk run behind it.
Evaluating $t(c)$ for every cut and taking the minimum handles all of this in a
few lines.

**Pinning.** While a request is served, its own blocks must not be chosen as
victims, in any tier: on the GPU they are about to be attended to, and in the lower
tiers they are about to be loaded. A consequence is the third branch of "place":
when a tier is full of pinned blocks, the block being placed passes through to the
next tier.

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

The index is updated in the same breath as each store call, so `_free` and
`_victim` always describe the store as it is. The linear scans are fine at these
sizes; a production index keeps one recency list per tier.

## Common mistakes

- **Evicting instead of demoting.** Dropping the GPU victim is the one-tier policy.
  It passes every test of a plain prefix cache and makes the lower tiers
  decoration: every returning prompt is prefilled from scratch.
- **Always loading.** "A stored block is a hit" ignores that a transfer can be
  slower than the prefill it replaces. The requests that suffer are the short
  ones, which then wait on a disk for something the GPU computes faster.
- **Deciding run by run.** Loading the host run only if it pays for itself, then
  the disk run only if it does, gives up a long disk run whenever a short host run
  in front of it is not worth its base cost alone. The comparison is between
  totals.
- **The wrong side of a tie.** With `<` the smaller cut wins; the statement gives
  ties to reuse.
- **Loading past a missing block.** Counting every stored block of the chain as
  reusable loads a block whose ancestor is gone. Its keys and values were computed
  after tokens the GPU no longer has; the store refuses the load.
- **A key without the parent.** The same tokens after another prefix are another
  block (the mistake of `block-hash-prefix-cache`, with the same consequence:
  one conversation is served another's keys and values).
- **Shallowest first.** Demoting the head of a chain before its tail breaks the
  ordering of tiers along a chain, and the next request for that prompt finds its
  first block on the slowest tier.
- **Not pinning.** Without it, making room for a request can push down the blocks
  the same request is about to load, or the ones it just counted as GPU hits.
- **No cascade.** Sending the GPU victim straight to disk when the host tier is
  full leaves old blocks in fast memory and puts recent ones on the slow tier.
- **Not refreshing on a hit.** If only loaded and computed blocks get a new stamp,
  the policy is first-in-first-out, and a system prompt that every request reuses
  on the GPU is the first thing to be demoted.
- **One transfer per block.** The base cost is paid once per transfer; loading
  block by block pays it every time.

## Complexity

Per call: $O(m)$ key computations, $O(m)$ cuts to evaluate ($O(m^2)$ as written,
with the counts recomputed for each cut), and a scan of the index per victim. Store
time is what the cut minimizes. Bytes moved are one block per block loaded and one
per demotion; a block that travels GPU to host to disk and back to the GPU is moved
three times.

## In the wild

What is established, from the projects' own material:

- **LMCache** (documentation, "Architecture Overview"; and the paper "LMCache: An
  Efficient KV Cache Layer for Enterprise-Scale LLM Inference", arXiv 2510.09665)
  describes this hierarchy for engines such as vLLM: GPU memory holds the active
  working set; CPU memory is a hot cache of recently used KV chunks, in pinned
  memory for fast GPU transfers; local storage is a larger tier; remote storage is
  persistent and slower. Its documented flows are the ones modeled here: overflow
  is offloaded from the GPU to CPU memory, written on from CPU memory to disk or
  remote storage under an LRU policy, and moved back to the GPU on demand. Its
  cache index cuts token sequences into chunks (256 tokens by default) and hashes
  them.
- **NVIDIA Dynamo's KV block manager** (KVBM design document) names the tiers G1
  (GPU device pool), G2 (host pool, pinned CPU memory), G3 (disk pool, local NVMe),
  and G4 (remote storage), and uses the verbs *offload* (device to host, host to
  disk) and *onboard* (host to device, and disk to device directly, without passing
  through the host). The direct disk-to-GPU load of this problem follows that.
- **vLLM**'s automatic prefix caching supplies the key scheme (a hash per full
  block chained to its parent), as in the prerequisite.

What is this problem's own simplification or choice:

- **A block lives in one tier.** Real stores usually keep the copy in the lower
  tier when a block is loaded, so demoting it again is free and a load does not
  empty a tier. Exclusive tiers make "bytes moved" a clean count and are why a
  full host tier of pinned blocks can occur at all.
- **Demotions take no time.** Writes to the lower tiers are asynchronous in real
  systems and off the request path; here they cost bytes and not latency. Loads
  and computations are serialized on one clock, and one request is served at a
  time.
- **The cost model and the cut rule.** `base + per_block * n` per transfer against
  a fixed time per computed block is a stand-in for bandwidth and prefill speed.
  The rule "minimize $t(c)$" is this problem's statement of the load-or-recompute
  trade; it is not a description of how LMCache or Dynamo decide.
- **The victim rule.** Least recently used with ties to the deepest block is a
  policy chosen here for its invariant; the projects above document LRU eviction
  between tiers but not this tie-break.
- **`discard`** stands in for anything that removes a block out of order (expiry,
  an administrator, a failed disk), which is the only way a stored block can end up
  behind a missing one under this policy.
