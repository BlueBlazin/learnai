Implement `RadixCache`, a token-level prefix cache backed by a radix tree. Every cached sequence is a path from the root, sequences that share a prefix share the same nodes, and each edge stores a non-empty run of tokens rather than a single token. Your implementation must split edges on partial matches, maintain lock counts along active paths, and evict the least recently used unlocked leaves on demand.

![This problem is Prefix lookup, here a radix tree over token sequences.](assets/map.svg)

## Specification

```python
class RadixCache:
    def __init__(self): ...
    def match_prefix(self, tokens): ...  # -> (length, node)
    def insert(self, tokens): ...  # -> int
    def lock(self, node): ...  # -> None
    def unlock(self, node): ...  # -> None
    def evict(self, n_tokens): ...  # -> int
    def total_tokens(self): ...  # -> int
    def nodes(self): ...  # -> list of (prefix, edge_length, locks)
```

**The tree.** The root represents the empty prefix. Every other node has an **edge** from its parent carrying one or more tokens, and represents the **prefix** formed by concatenating all edge tokens from the root down to that node. No two children of the same node have edges starting with the same token, so any token sequence follows at most one path. Each non-root node has a **lock count** (initially 0) and a **last access time**. The cache stores no tensors. In an inference engine, each node would own the KV entries for the tokens on its edge.

**The clock.** The cache uses a logical clock that starts at 0. Every call to `match_prefix` or `insert` increments the clock by 1 at the start of the call. No other method advances the clock.

**The walk.** Both `match_prefix` and `insert` begin with the same tree walk. Starting at the root, follow `tokens` down the tree for as long as they match. The **matched length** $L$ is the length of the longest prefix of `tokens` that is a prefix of some node's prefix in the tree. If the walk ends partway through an edge (meaning the first $c \ge 1$ tokens of the edge matched, and more tokens remain on that edge), **split** the edge after $c$ tokens:

- Create a new *upper* node carrying the first $c$ tokens and attach it under the old parent. The existing node keeps the remaining tokens, its children, its lock count, its last access time, and its object identity, and becomes the upper node's only child.
- The upper node **inherits** the lock count and the last access time of the existing node.

After the walk, a node ends at exact length $L$ (the root if $L = 0$). Finally, update the last access time of every node on the path from the root to that node (including the upper half of any split) to the current clock value. The lower half of a split edge was not matched, so it is not on that path and keeps its previous access time.

- `match_prefix(tokens)` runs the walk and returns `(L, node)`. Here `node` is a **handle** to the node ending at length $L$. It can be any object you choose, and callers pass it to `lock` and `unlock`. A handle must continue referring to the same prefix even if edges above or below it are split later, and it remains valid until its node is evicted.
- `insert(tokens)` runs the walk. If $L < \text{len}(\text{tokens})$, it adds one new leaf under the node ending at length $L$ with edge `tokens[L:]`, lock count 0, and the current clock value. It returns $L$, the number of tokens that were already cached. If `tokens` is already a prefix in the tree, the walk may still split an edge, and no new leaf is added.
- `lock(node)` increments the lock count of `node` and every ancestor except the root by 1. `unlock(node)` decrements the lock count of those same nodes by 1. On the root's handle, both methods do nothing. Both return `None`. Tests only unlock a handle as many times as it was locked.
- `evict(n_tokens)` removes nodes until at least `n_tokens` tokens have been evicted or no evictable node remains. At each step, consider all **leaves** (non-root nodes with no children) whose lock count is 0, remove the leaf with the smallest last access time, and add its edge length to the removed token count. If removing a leaf leaves its parent with no children, that parent becomes a leaf immediately and is eligible in the same `evict` call. Inner nodes are never removed, and edges are never merged. Return the total number of tokens removed. This is 0 when `n_tokens = 0`, and may exceed `n_tokens` because each leaf is removed whole.
- `total_tokens()` returns the total number of tokens across all edges in the tree.
- `nodes()` returns a sorted list of `(prefix, edge_length, locks)` tuples, one for each non-root node. `prefix` is the node's full prefix as a tuple of ints, `edge_length` is the number of tokens on its own edge, and `locks` is its lock count.

`tokens` is a list of ints and may be empty. Do not modify `tokens`.

> [!NOTE]
> No two leaves ever share the same last access time. Each access stamps nodes along a single path from the root, and any path contains at most one leaf. The eviction order is therefore always uniquely determined.

## Example

```python
cache = RadixCache()
cache.insert([1, 2, 3, 4, 5, 6])  # 0    time 1
cache.insert([1, 2, 3, 7, 8])  # 3    time 2: the edge is split after 1 2 3
n, h = cache.match_prefix([1, 2, 3, 4, 9])  # n = 4  time 3: 4 5 6 is split after 4
cache.lock(h)
cache.insert([9, 9, 9])  # 0    time 4
cache.nodes()
# [((1, 2, 3), 3, 1), ((1, 2, 3, 4), 1, 1), ((1, 2, 3, 4, 5, 6), 2, 0),
#  ((1, 2, 3, 7, 8), 2, 0), ((9, 9, 9), 3, 0)]
```

```
root ─┬─ [1 2 3] ─┬─ [4] ─── [5 6]        times: [1 2 3] 3, [4] 3, [5 6] 1,
      │   lock 1  │  lock 1                      [7 8] 2, [9 9 9] 4
      │           └─ [7 8]
      └─ [9 9 9]
```

The second insert shares three tokens with the first sequence. The edge `1 2 3 4 5 6` splits into `1 2 3` and `4 5 6`, and `7 8` is attached as a second child. Next, `match_prefix([1, 2, 3, 4, 9])` ends one token into `4 5 6`, splitting that edge as well. The returned handle `h` refers to node `[4]` (prefix `1 2 3 4`). The unmatched lower half `[5 6]` keeps time 1. Calling `lock(h)` locks both `[4]` and its ancestor `[1 2 3]`.

```python
cache.evict(4)  # 4    removes [5 6] (time 1), then [7 8] (time 2)
cache.evict(100)  # 3    removes [9 9 9]; [4] is a leaf now, but locked
cache.unlock(h)
cache.evict(1)  # 1    removes [4]
cache.total_tokens()  # 3    only [1 2 3] is left
cache.evict(1)  # 3    a leaf goes whole
```

![The tree at four points of the example. Each box is a node, labeled with the tokens on its edge.](assets/tree-steps.svg)

## Constraints

- Input sequences have up to a few hundred tokens, each cache receives at most a few hundred calls, and token IDs are non-negative ints.
- All outputs are checked for exact equality after every operation. The tests verify the return value, the full `nodes()` list (checking every split, lock count, and eviction), and `total_tokens()`.
- **Property check.** On random workloads, `match_prefix` is validated against a brute-force reference storing the set of cached prefixes. The matched length must equal the longest common prefix of `tokens` with any prefix in that set.
- Handles are locked and unlocked only while their node exists in the tree.

## Why this matters

Fixed-block hashing finds shared prefixes only up to the last aligned block boundary. A radix tree matches down to the exact token. A multi-turn chat, an agent retrying with a different tool output, and few-shot prompts sharing examples all reuse every common token.

This data structure powers SGLang's RadixAttention and runs on the scheduler's hot path. Its three core invariants correspond directly to three severe failure modes. Failing to split an edge on a partial match either causes a cache miss or hands a request a node whose KV cache it only partly owns. Miscounting locks lets the evictor free a prefix that a running request is still attending to. Evicting an inner node or the wrong leaf destroys a widely shared system prompt while retaining one-off suffixes underneath it.
