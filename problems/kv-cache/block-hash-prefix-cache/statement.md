Implement `PrefixBlockCache`, the block-level index behind automatic prefix caching. Each token sequence is split into fixed-size blocks, and each full block gets a key that identifies its entire prefix. Because a token's attention keys and values depend on all preceding tokens, two prompts can share a cached block only when their prefixes agree up to the end of that block.

![This problem is Prefix lookup and the Block pool, where a prompt's leading blocks are found by their keys.](assets/map.svg)

## Specification

```python
class PrefixBlockCache:
    def __init__(self, block_size, capacity): ...
    def lookup(self, tokens): ...  # -> int
    def acquire(self, request_id, tokens): ...  # -> (hits, held)
    def release(self, request_id): ...  # -> None
    def ref_count(self, tokens): ...  # -> int
    def cached_blocks(self): ...  # -> int
```

The cache stores metadata only and holds no tensors. It tracks which blocks are cached, how many active requests hold each block, and which idle block to evict when space is needed.

**Blocks.** Let $B = \text{block\_size}$ and let $n = \text{len}(\text{tokens})$. Block $k$ of `tokens` is the slice `tokens[k*B : (k+1)*B]` for $0 \le k < \lfloor n / B \rfloor$. Only **full** blocks are cached. The remaining $n \bmod B$ trailing tokens are ignored by every method.

**Identity.** Block $k$ of one token list and block $k'$ of another represent the same cached block if and only if $k = k'$ and both lists match on **all** tokens through the end of the block (`tokens[: (k+1)*B]`). To represent the whole prefix while spending only $O(B)$ work per block, chain each block's key with its parent's key:

$$
\text{key}_k = H(\text{key}_{k-1},\ \text{tokens}[kB : (k+1)B]), \qquad \text{key}_{-1} = \text{a constant}.
$$

Any scheme that satisfies this identity is accepted, provided two different prefixes do not collide in practice (for example, a 64-bit or wider hash, or exact keys).

**State.** The cache holds at most `capacity` blocks. Each cached block has a **reference count** equal to the number of active requests holding it. Cached blocks with a reference count of 0 form the **eviction order**, which behaves as a queue. A block joins the back of the queue when its reference count drops to 0, and leaves the queue when it is referenced again or evicted.

- `lookup(tokens)` returns the number of **leading** blocks of `tokens` currently in the cache. That is the largest $h$ such that blocks $0, \ldots, h - 1$ are all cached. It does not modify any reference count or the eviction order.

- `acquire(request_id, tokens)` starts a request for `tokens` and returns `(hits, held)`:

  1. Compute `hits = lookup(tokens)`. Increment the reference count of each of those `hits` blocks by 1, and remove any of them whose count was 0 from the eviction order.
  2. For each remaining full block of `tokens` in order, check whether the cache already holds `capacity` blocks. If it does and the eviction order is empty, stop immediately so this block and all later ones are not cached. If it does and the eviction order is non-empty, evict (remove) the block at the **front** of the eviction order. Then insert the new block with a reference count of 1.
  3. Set `held` to `hits` plus the number of blocks inserted. The request holds blocks $0, \ldots, \text{held} - 1$ of `tokens` until it is released.

  Step 1 must complete before step 2 begins. If `request_id` is already active, raise `ValueError` without changing the cache.

- `release(request_id)` ends an active request. Decrement the reference count of each held block in reverse order, **deepest block first** (block `held - 1` down to block 0). Whenever a block's reference count reaches 0, append it to the back of the eviction order at that moment. Return `None`. Raise `KeyError` if `request_id` is not active.

- `ref_count(tokens)` returns the reference count of the **last** full block of `tokens` (block $\lfloor n / B \rfloor - 1$). Raise `KeyError` if `tokens` has no full block or if that block is not cached.

- `cached_blocks()` returns the number of blocks currently in the cache.

`request_id` can be any hashable value. `tokens` is a list of ints and may be empty. Do not modify `tokens`.

> [!NOTE]
> Releasing the deepest block first places each block ahead of its parent in the eviction order. Together with completing step 1 before step 2 in `acquire`, this keeps the cache *prefix-closed*. Every cached block's parent is also cached, so no cache slot is ever wasted on a block that `lookup` cannot reach.

## Example

`block_size = 4`, `capacity = 5`. `S` is a system prompt of two blocks.

```python
S = [1, 2, 3, 4, 5, 6, 7, 8]
Q1 = S + [11, 12, 13, 14, 15, 16]  # 3 full blocks, 2 tokens left over
Q2 = S + [21, 22, 23, 24, 25]  # 3 full blocks, 1 token left over
T = [31, 32, 33, 34, 35, 36, 37, 38, 39]
X = [41, 42, ..., 52]  # 12 tokens, 3 full blocks
Y = [61, 62, ..., 68]  # 8 tokens, 2 full blocks

cache = PrefixBlockCache(block_size=4, capacity=5)
cache.acquire("r1", Q1)  # (0, 3)   S0, S1, Q1's block 2 inserted
cache.lookup(S + [99])  # 2        the two blocks of S
cache.lookup([5, 6, 7, 8, 1, 2, 3, 4])  # 0        S's blocks swapped: different prefixes
cache.acquire("r2", Q2)  # (2, 3)   S0, S1 reused; Q2's block 2 inserted
cache.release("r1")  #          eviction order: [Q1.2]  (S0, S1: count 1)
cache.acquire("r3", T)  # (0, 2)   T0 fills the cache; T1 evicts Q1.2
cache.lookup(Q1)  # 2
cache.release("r2")  #          eviction order: [Q2.2, S1, S0]
cache.acquire("r4", X)  # (0, 3)   evicts Q2.2, S1, S0, in that order
cache.lookup(Q2)  # 0
cache.acquire("r5", Y)  # (0, 0)   5 blocks, all referenced: nothing to evict
cache.release("r5")
cache.release("r3")  #          eviction order: [T1, T0]
cache.acquire("r6", Y)  # (0, 2)   evicts T1, then T0
cache.cached_blocks()  # 5
```

![`Q1` and `Q2` as chains of block keys. They share $\text{key}_0$ and $\text{key}_1$, then differ in block 2.](assets/key-chain.svg)

The third call shows why each key must include its parent key. Both four-token blocks of `[5, 6, 7, 8, 1, 2, 3, 4]` appear in `S`, but block 0 does not match `S`'s block 0, and block 1 follows a different prefix. Neither block is a hit.

`release("r2")` appends `Q2`'s third block first and the first block of `S` last. When `r4` needs three slots, the cache evicts the request-specific tail first and the shared head of the system prompt last.

## Constraints

- $1 \le \text{block\_size} \le 64$, $1 \le \text{capacity} \le 4096$, prompts have up to 32,000 tokens, token IDs are ints in $[0, 2^{31})$, and each cache receives at most a few hundred calls.
- All outputs are checked for exact equality. After every operation, the tests verify its return value and probe the cache using the read-only methods. They call `lookup` on every token list seen so far, `ref_count` on every block ever cached (expecting `KeyError` for evicted blocks), and `cached_blocks()`. Probing must not alter cache state.
- Keying and looking up a prompt of 2,000 blocks must finish well under a second.

## Why this matters

Real LLM workloads repeat prefixes constantly, including shared system prompts, tool definitions, and multi-turn chat histories. With this index, an inference server skips prefill for every prefix it has already processed without requiring callers to label shared segments, which is why it is called *automatic* prefix caching.

vLLM uses this exact design with a parent-chained hash per full block, a reference count per block, and a free queue for eviction. Breaking any rule in this specification causes a silent production bug. Dropping the parent key serves one conversation's KV cache to another that happens to share a block of tokens. Caching a partial block lets a second request read stale slots while the first request is still writing them. Evicting a referenced block overwrites KV memory that an active request is still attending to. Finally, a hash that can collide leaks data across tenants, which is why vLLM switched its default block hash to SHA-256 in v0.11.
