## The idea

Because attention is causal, the keys and values computed for token $i$ depend on tokens $0$ through $i$ and nothing else. Any new request can therefore reuse a cached KV block as long as its prompt matches the original **up to the end of that block**.

**Keying a prefix with chained hashes.** Let $B = \text{block\_size}$ and let $n$ be the number of tokens in the prompt. Keying a block by the tuple of its entire prefix is correct, but it costs $O(n)$ per block and $O(n^2 / B)$ per prompt. Chaining the keys reduces the cost to $O(B)$ per block. For each block $k$ (starting from a constant $\text{key}_{-1}$):

$$
\text{key}_k = H(\text{key}_{k-1},\ \text{block}_k).
$$

By induction, $\text{key}_k$ depends on every token through block $k$. With a collision-resistant hash $H$, distinct prefixes produce distinct keys. Computing each block's key hashes only $B$ tokens plus a fixed-size parent key. This is a Merkle chain, the same construction used for Git commit IDs and blockchain headers.

**Lookup is a prefix walk.** Compute the keys of the prompt's full blocks and count how many leading keys exist in the cache. The walk stops at the first miss. No later block can be a hit because its key incorporates the missing block's key.

**Only full blocks are cached.** A partially filled block is still being written by its owning request. If another request shared it, that request would read or overwrite slots belonging to a different continuation. The trailing $n \bmod B$ tokens (at most $B - 1$) are therefore always recomputed, which means cache hits always occur in multiples of $B$.

**Reference counts and eviction order.** A block held by an active request cannot be reclaimed, so each block tracks how many active requests reference it. Blocks with a reference count of 0 stay in the cache in case their prefix appears again, and they are evicted in the order they became idle. Two ordering rules keep the cache consistent:

- **Reference hits before evicting.** Incrementing reference counts for hit blocks before inserting new ones prevents a request from evicting the very blocks it is about to reuse.
- **Release the deepest block first.** Releasing in reverse order places every child ahead of its parent in the eviction queue. The cache therefore stays prefix-closed, meaning no cached block ever has an evicted parent. Without this rule, the head of a chain would be evicted first and leave its descendants unreachable while still occupying cache slots. This order also matches reuse patterns, since the tail of a prompt is request-specific while the head (such as a system prompt) is shared widely.

## Reference solution (Python)

```python
import hashlib
import struct
from collections import OrderedDict


class PrefixBlockCache:
    def __init__(self, block_size, capacity):
        self.block_size = block_size
        self.capacity = capacity
        self._refs = {}  # key -> reference count
        self._evictable = OrderedDict()  # keys with count 0, least recently released first
        self._held = {}  # request id -> its keys

    def _keys(self, tokens):
        size = self.block_size
        keys, parent = [], b""
        for start in range(0, len(tokens) - size + 1, size):
            block = struct.pack(f"<{size}q", *tokens[start : start + size])
            parent = hashlib.sha256(parent + block).digest()
            keys.append(parent)
        return keys

    def _hits(self, keys):
        hits = 0
        while hits < len(keys) and keys[hits] in self._refs:
            hits += 1
        return hits

    def lookup(self, tokens):
        return self._hits(self._keys(tokens))

    def acquire(self, request_id, tokens):
        if request_id in self._held:
            raise ValueError(f"request {request_id!r} is already active")
        keys = self._keys(tokens)
        hits = self._hits(keys)
        for key in keys[:hits]:
            self._refs[key] += 1
            self._evictable.pop(key, None)
        held = hits
        for key in keys[hits:]:
            if len(self._refs) == self.capacity:
                if not self._evictable:
                    break
                victim, _ = self._evictable.popitem(last=False)
                del self._refs[victim]
            self._refs[key] = 1
            held += 1
        self._held[request_id] = keys[:held]
        return hits, held

    def release(self, request_id):
        for key in reversed(self._held.pop(request_id)):
            self._refs[key] -= 1
            if self._refs[key] == 0:
                self._evictable[key] = None

    def ref_count(self, tokens):
        keys = self._keys(tokens)
        if not keys:
            raise KeyError("tokens hold no full block")
        return self._refs[keys[-1]]

    def cached_blocks(self):
        return len(self._refs)
```

Python's built-in `hash((parent, tuple(block)))` is also accepted here because it is 64 bits wide and chains the same way. Using a nested tuple `(parent_tuple, block_tuple)` directly as a dictionary key is exact, but Python must rehash the entire nested structure on every lookup, which makes it too slow for long prompts.

## Common mistakes

- **Hashing the block's tokens alone.** With `key = H(block)`, a block matches regardless of what preceded it. After inserting `a b` and `c d`, the prompt `a d` reports two hits, causing the server to attend to keys and values computed after `c`. In production this produces silent corruption that depends on other users' traffic.
- **Caching the partial last block.** Including the trailing partial block makes `lookup` report one block too many for an identical prompt and exposes a block that its owner is still writing.
- **Evicting a referenced block.** Running a plain LRU over all blocks ignores reference counts and hands an active request's KV memory to another request.
- **Using a boolean flag instead of a count.** Marking a block as "in use" and clearing the flag on release frees the block as soon as the *first* of several active holders finishes.
- **Leaving a hit in the eviction queue.** Incrementing a block's reference count without removing it from the eviction queue leaves an active block eligible for eviction.
- **Referencing hits after inserting new blocks.** Making room before incrementing hit counts can evict the very blocks the request is about to reuse, leaving the request holding missing blocks.
- **Releasing the first block first.** Queuing block 0 before deeper blocks puts the head of a shared prefix at the front of the eviction queue. After one eviction, the remaining blocks in the chain are unreachable yet still consume cache slots.
- **Using FIFO instead of least-recently-released eviction.** Evicting by insertion order discards the oldest shared blocks even when they have been reused continuously.
- **Giving `lookup` side effects.** Updating eviction order or counts during `lookup` turns a read-only routing query into a state mutation.
- **Growing past `capacity` or never evicting.** When all cached blocks are referenced, the cache must stop inserting new blocks. When idle blocks exist, it must evict from the front of the eviction queue.

## Complexity

Keying a prompt of $n$ tokens takes $O(n)$ time. After keying, `lookup`, `acquire`, and `release` each take $O(n / B)$ time with $O(1)$ work per block. Memory usage is $O(\text{capacity})$, storing one key and one reference count per cached block.

## In the wild

- **vLLM automatic prefix caching** (`docs/design/prefix_caching.md`). vLLM hashes each KV block from its parent block's hash, its own token IDs, and extra metadata (such as LoRA ID, image hashes, or an optional per-request `cache_salt`). It caches only full blocks, and since v0.11 it uses SHA-256 by default to prevent hash collisions.
- **The free queue.** Each `KVCacheBlock` stores a `ref_cnt`. Blocks with `ref_cnt == 0` sit in a doubly linked free queue while retaining their hash. When a request hits cached blocks, vLLM increments their counts and removes them from the free queue before popping new blocks from the front. On release, blocks are appended in reverse order so tail blocks are evicted before shared prefix heads (`KVCacheManager.free` in `vllm/v1/core/kv_cache_manager.py`).
- **Differences from this problem.** In vLLM, the free queue manages physical GPU blocks, so both unused blocks and idle cached blocks sit in the same pool bounded by GPU memory. A request that cannot obtain enough blocks waits to be scheduled instead of running partially uncached. Blocks also enter the cache as decoding fills them, not only during prefill.
- **The alternative.** SGLang indexes prefixes with a radix tree instead of per-block hashes, matching at token granularity rather than fixed block boundaries (`radix-prefix-cache`).
