## The idea

The keys and values a model computes for token $i$ depend on tokens $0$ to $i$
and on nothing else. So a block of KV cache can be reused by any request whose
prompt agrees with the original **up to the end of that block**. That sentence is
the whole specification of a prefix cache; the rest is making it cheap.

**A key for a prefix in $O(B)$.** Storing a block under the tuple of its whole
prefix is correct and costs $O(n)$ per block, $O(n^2 / B)$ per prompt. Chain the
keys instead:

$$
\text{key}_k = H(\text{key}_{k-1},\ \text{block}_k).
$$

By induction, $\text{key}_k$ is a function of all tokens through block $k$, and
with a collision-resistant $H$ two different prefixes get different keys. Each
block costs one hash of $B$ tokens plus a fixed-size parent key. This is a Merkle
chain, the construction of a git commit id or a blockchain header.

**Lookup is a walk.** Compute the keys of the prompt's full blocks and count how
many leading ones are in the cache. The walk stops at the first miss: a later
block cannot be a hit, because its key contains the missing block's key.

**Only full blocks.** A partly filled block is still being written by the request
that owns it. If it were shared, the next request would read slots that belong to
someone else's continuation (or write its own). So the tail of the prompt, up to
$B - 1$ tokens, is always recomputed. That is the price of the fixed block grid,
and the reason hits come in multiples of the block size.

**Reference counts and the eviction order.** A block in use by a running request
must not be reclaimed, so every block counts its holders. Blocks with count 0 stay
cached, in case the prefix returns, and are evicted in the order in which they
became idle. Two details keep the structure sound:

- *Hits are referenced before room is made.* Otherwise inserting a request's new
  blocks could evict the very blocks it is about to reuse.
- *A release queues the deepest block first.* A child then always sits ahead of
  its parent in the eviction order, so the cache stays prefix-closed: no cached
  block has an evicted parent. Without this, the head of a chain would be evicted
  first and the rest would occupy slots that no lookup can reach. It is also the
  right policy: the tail of a prompt is specific to one request, the head (the
  system prompt) is what everyone shares.

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

`hash((parent, tuple(block)))` with Python's built-in hash is also accepted here:
it is 64 bits wide and chains the same way. A nested tuple
`(parent_tuple, block_tuple)` used directly as the dict key is exact but slow,
because Python rehashes the whole nesting on every lookup.

## Common mistakes

- **Hashing the block's tokens alone.** The mistake the design exists to prevent.
  With `key = H(block)` a block is found wherever those tokens occur: after
  inserting `a b` and `c d`, the prompt `a d` reports two hits, and the server
  would attend to keys and values computed after `c`. In production this is
  silent corruption that depends on what other users sent.
- **Caching the partial last block.** It makes `lookup` report one block too many
  for an identical prompt, and hands out a block that its owner is still filling.
- **Evicting a referenced block.** A plain LRU over all blocks ignores the counts.
  The running request's block table then points at memory that has been given to
  someone else.
- **A flag instead of a count.** Marking a block "in use" and clearing the mark on
  release frees it when the *first* of several holders finishes.
- **A hit that stays in the eviction order.** Incrementing the count without
  removing the block from the queue leaves a referenced block at the front of it.
- **Referencing the hits after inserting.** Making room first can evict the blocks
  the request is about to reuse; the request then holds blocks that no longer
  exist.
- **Releasing first block first.** The system prompt's first block reaches the
  front of the eviction order ahead of everything that depends on it. One eviction
  later the rest of the chain is unreachable and still occupies the cache.
- **FIFO instead of least recently released.** Evicting by insertion time punishes
  exactly the blocks that have proven useful the longest.
- **A `lookup` with side effects.** Refreshing a block's position on lookup turns
  a query (a router asking "how much of this would you have?") into a use.
- **Growing past the capacity**, or **never evicting**. When every block is
  referenced the right answer is to cache nothing more; when some are idle, the
  oldest idle one goes.

## Complexity

Keying a prompt is $O(n)$; `lookup` is $O(n / B)$ dictionary probes after that;
`acquire` and `release` are $O(n / B)$ with $O(1)$ work per block. Memory is one
key and one count per cached block.

## In the wild

- **vLLM, automatic prefix caching** (`docs/design/prefix_caching.md`): "we hash
  each kv-cache block by the tokens in the block and the tokens in the prefix
  before the block". The hash covers the parent block's hash, the block's token
  ids, and extra keys (LoRA id, image hashes, an optional per-request
  `cache_salt`). The document states "We only cache full blocks", and that as of
  v0.11 the default hash is SHA-256 because earlier keys were not guaranteed to
  be collision-free.
- **The same queue.** Each `KVCacheBlock` has a `ref_cnt`; blocks with
  `ref_cnt == 0` sit in a doubly linked free queue and keep their hash. A new
  request's hit blocks are "touched" (count incremented, removed from the queue)
  before new blocks are popped from the head of the queue, and popping a cached
  block evicts it. On release, blocks are appended "in the *reverse* order ...
  because the last block of a request must hash more tokens and is less likely to
  be reused" (the design document; `KVCacheManager.free` in
  `vllm/v1/core/kv_cache_manager.py` says "We free the blocks in reverse order so
  that the tail blocks are evicted first").
- **Differences from this problem.** In vLLM the queue holds physical blocks, so
  a never-used block and an idle cached block are both "free", and the capacity
  is the GPU's block pool. A request that cannot get its blocks is not scheduled,
  rather than run uncached. Blocks also become cached as decoding fills them, not
  only at prefill.
- **The alternative.** SGLang indexes prefixes with a radix tree instead of a
  hash per block, which matches at token rather than block granularity
  (`radix-prefix-cache`).
