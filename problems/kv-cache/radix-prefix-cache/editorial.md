## The idea

A prefix cache answers one question: how many leading tokens of a prompt already have cached KV entries? A token-level trie answers this by walking from the root so that sequences with a common prefix share nodes and store shared KV entries once. A plain trie allocates one node per token, which is too large for prompts of thousands of tokens. A **radix tree** compresses every chain of single-child nodes into a single edge carrying the entire run of tokens. The number of nodes then scales with the number of branch points and sequence ends rather than total tokens.

Compression introduces one complication. A new sequence can diverge from or end in the middle of an existing edge. When that happens, you must **split** the edge at the divergence point because a node is the unit of locking, eviction, and branching.

Three invariants keep the tree sound:

1. **Children of a node start with distinct tokens.** This makes the walk deterministic with a single dictionary lookup per node. Splitting restores this invariant whenever a new sequence shares part of an existing edge.
2. **A lock covers the entire path to the root.** A running request reusing the prefix at node $x$ depends on the KV entries of $x$ and all of its ancestors. Incrementing the lock count along the whole path makes checking whether a node is in use a local test (`locks > 0`). During a split, the new upper half sits above everything that locked the old edge, so it inherits the lock count. A later `unlock` walks up from the handle through the upper half and decrements it properly.
3. **Only leaves are evicted.** An inner node is a prefix of its descendants, whose KV entries depend on it and become unreachable without it. Eviction therefore works from the outside in, removing the least recently used unlocked leaf first. When a parent loses its last child, it becomes a leaf and joins the eviction candidates. Because every access stamps the entire path, a parent is never older than its children, allowing a single `evict` call to reclaim a cold branch from tip to root.

## Reference solution (Python)

```python
import heapq


class Node:
    def __init__(self, parent, key, last_access):
        self.parent = parent
        self.key = key
        self.children = {}
        self.locks = 0
        self.last_access = last_access


class RadixCache:
    def __init__(self):
        self._root = Node(None, [], 0)
        self._clock = 0
        self._tokens = 0

    def _walk(self, tokens):
        self._clock += 1
        node, matched = self._root, 0
        while matched < len(tokens):
            child = node.children.get(tokens[matched])
            if child is None:
                break
            key = child.key
            common = 1
            limit = min(len(key), len(tokens) - matched)
            while common < limit and key[common] == tokens[matched + common]:
                common += 1
            if common < len(key):
                child = self._split(child, common)
            child.last_access = self._clock
            node, matched = child, matched + common
            if common < len(key):
                break
        return matched, node

    def _split(self, child, length):
        upper = Node(child.parent, child.key[:length], child.last_access)
        upper.locks = child.locks
        child.parent.children[upper.key[0]] = upper
        child.key = child.key[length:]
        child.parent = upper
        upper.children[child.key[0]] = child
        return upper

    def match_prefix(self, tokens):
        return self._walk(tokens)

    def insert(self, tokens):
        matched, node = self._walk(tokens)
        if matched < len(tokens):
            leaf = Node(node, list(tokens[matched:]), self._clock)
            node.children[leaf.key[0]] = leaf
            self._tokens += len(leaf.key)
        return matched

    def lock(self, node):
        while node is not self._root:
            node.locks += 1
            node = node.parent

    def unlock(self, node):
        while node is not self._root:
            node.locks -= 1
            node = node.parent

    def evict(self, n_tokens):
        heap = []
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.children:
                stack.extend(node.children.values())
            elif node is not self._root and node.locks == 0:
                heap.append((node.last_access, id(node), node))
        heapq.heapify(heap)
        evicted = 0
        while evicted < n_tokens and heap:
            _, _, victim = heapq.heappop(heap)
            parent = victim.parent
            del parent.children[victim.key[0]]
            evicted += len(victim.key)
            if not parent.children and parent is not self._root and parent.locks == 0:
                heapq.heappush(heap, (parent.last_access, id(parent), parent))
        self._tokens -= evicted
        return evicted

    def total_tokens(self):
        return self._tokens

    def nodes(self):
        out = []
        stack = [(self._root, ())]
        while stack:
            node, prefix = stack.pop()
            for child in node.children.values():
                path = prefix + tuple(child.key)
                out.append((path, len(child.key), child.locks))
                stack.append((child, path))
        return sorted(out)
```

`_split` creates a new node for the upper half and reuses the existing node as the lower half. Doing the reverse would keep the tree shape valid, but any handle previously returned for the existing node would silently change meaning from the full prefix to the shorter upper prefix. The `id(node)` tie-breaker in the heap only prevents Python from comparing `Node` objects, since leaf access times never tie.

## Common mistakes

- **Matching only whole nodes.** Skipping splits in `match_prefix` wastes partial matches, such as sharing 900 tokens of a 1,000-token edge. Skipping splits in `insert` attaches a second child starting with the same token, which overwrites its sibling in the `children` dict and drops an entire subtree from lookups while still counting its tokens.
- **Evicting an inner node.** Evicting the least recently used unlocked node instead of restricting to leaves removes a shared prefix along with everything beneath it, or unlinks it while leaving the subtree's KV memory allocated. It also misreports the evicted token count by counting only the node's own edge instead of its descendants.
- **Not stamping ancestors.** Updating only the final matched node leaves inner nodes with their creation timestamps. Those nodes stay safe while they have children, but once their children are evicted they appear older than leaves that have been idle much longer. This bug only surfaces when a locked bystander forces eviction across branches, so simple tests often miss it.
- **Stamping a partly matched node.** Setting `last_access` upon entering an edge, before checking how many tokens match, refreshes the unmatched lower half of a split and makes unused tokens appear recently accessed.
- **Dropping the lock count during a split.** Starting the new upper half at 0 ignores the active requests holding the lower half. Calling `unlock` then decrements the upper half to -1, and in a real engine tracking evictable memory by lock count, the accounting is off by the upper half's tokens.
- **Locking only the target node.** Failing to lock ancestors causes the same accounting error in reverse. Worse, after a split, an inherited lock count on the upper half is never decremented on `unlock`, permanently pinning that branch.
- **Skipping the eviction cascade.** If a parent that becomes a leaf is not pushed onto the heap, `evict` frees fewer tokens than requested even when more evictable nodes remain, forcing the scheduler to fail or retry.
- **Off-by-one or comparison bugs.** Ignoring `locks`, popping the newest leaf instead of the oldest, or looping while `evicted <= n_tokens` instead of `evicted < n_tokens` each break eviction with a single wrong condition.

## Complexity

Let $n$ be the input sequence length, $d$ be the number of nodes along its path, and $N$ be the total number of nodes in the tree. The walk performs $O(n)$ token comparisons and $O(d)$ dictionary lookups. A split takes $O(\text{edge length})$ time to slice the token list. Both `lock` and `unlock` run in $O(d)$ time. `evict` takes $O(N)$ time to collect the initial leaves plus $O(\log N)$ per removed node. Total memory is $O(\text{total tokens})$. Production engines maintain the set of evictable leaves incrementally instead of scanning the tree on each eviction.

## In the wild

- **The paper.** Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (NeurIPS 2024), introduced RadixAttention. It keeps the KV cache of finished requests in a radix tree keyed by tokens, with LRU leaf eviction and reference counts on nodes used by active requests.
- **The implementation** (`python/sglang/srt/mem_cache/radix_cache.py`). `match_prefix` walks the tree and calls `_split_node` when a match ends inside an edge. `_split_node` creates the new upper node, copies `new_node.lock_ref = child.lock_ref`, and keeps `child` as the lower half. `inc_lock_ref` and `dec_lock_ref` walk upward `while node != self.root_node`. `evict` builds a heap of evictable leaves ordered by priority (last access time under LRU), deletes each popped leaf, and pushes its parent when `len(x.parent.children) == 0 and x.parent.lock_ref == 0`.
- **Differences from this problem.** (1) SGLang stamps `child.last_access_time` as soon as the walk enters a child, before measuring the match, so both halves of a split receive the new timestamp. This problem stamps only fully matched nodes, which reflects actual reuse more strictly and guarantees unique leaf timestamps. (2) SGLang uses `time.monotonic()` instead of a logical counter. (3) With a paged allocator, SGLang aligns matches and splits to page-size multiples and stores KV slot indices on each node. (4) Alternative eviction policies such as LFU or priority plug into the same heap.
- **Comparison with block hashing.** vLLM's index (`block-hash-prefix-cache`) uses a flat hash table with $O(1)$ probes per block and hits at block-size multiples. A radix tree matches to the exact token and represents shared structure explicitly, at the cost of pointer traversal and edge splits. Both systems evict from the tail of a prefix inward for the same reason.
