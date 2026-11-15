## The idea

A prefix cache answers one question: *how many leading tokens of this prompt do I
already have KV entries for?* A trie over tokens answers it by walking, and
sequences with a common prefix share the nodes of that prefix, so the shared KV
entries are stored once. A plain trie has one node per token, which is hopeless
for prompts of thousands of tokens. A **radix tree** compresses every chain of
single-child nodes into one edge that carries the whole run. The number of nodes
is then proportional to the number of *branch points and sequence ends*, not to
the number of tokens.

Compression has a price: a new sequence can diverge from, or end in, the middle of
an edge. Then the edge must be **split** at that point, because a node is the unit
of everything else: what a request locks, what the evictor removes, and where a
new branch can attach.

Three invariants carry the design.

1. **Children of a node start with different tokens.** That makes the walk
   deterministic, a dict lookup per node, and it is what a split restores when a
   new sequence shares part of an edge.
2. **A lock covers a whole path.** A running request that reuses the prefix ending
   at node $x$ depends on the KV entries of $x$ and of every ancestor. Incrementing
   the count on the whole path makes "this node is in use" a local test
   (`locks > 0`). Splitting must preserve it: the new upper half is above
   everything that held a lock on the old edge, so it inherits the count, and a
   later `unlock`, which walks up from the handle, passes through it and releases
   it.
3. **Only leaves are evicted.** An inner node is a prefix of its descendants;
   their KV entries were computed attending to it, and are unreachable without it.
   Eviction therefore eats the tree from the outside in: least recently used
   unlocked leaf first, and a parent becomes a candidate once its last child is
   gone. Because an access stamps the whole path, a parent is never older than
   its children, and the cascade removes a cold branch tip to root in one call.

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

The split creates the upper half and keeps the existing node as the lower half.
The other way round also works for the tree, but then every handle given out for
the old node would silently change meaning from "the prefix ending here" to "the
first part of it". The `id(node)` in the heap entries only keeps Python from
comparing two nodes; times never tie between leaves.

## Common mistakes

- **Matching only whole nodes.** Without the split in `match_prefix`, a prompt
  that shares 900 of the 1,000 tokens on an edge reuses nothing. With whole-edge
  matching in `insert` and no split, the new sequence is attached as a second
  child that starts with the same token as its sibling and replaces it in the
  children dict: an entire subtree disappears from lookups while its tokens are
  still counted.
- **Evicting an inner node.** "Least recently used unlocked node" instead of
  "leaf" removes a prefix together with everything below it, or worse, unlinks it
  and leaves the subtree's KV entries allocated. The count returned is also wrong:
  the node's own tokens are counted, the subtree's are not.
- **Not stamping the ancestors.** If only the node where the match ends gets the
  new time, an inner node keeps the time of its creation. It is protected while
  it has children, but once they are evicted it looks like the oldest leaf in the
  tree and goes before leaves that were idle far longer. It takes a locked
  bystander to observe (see the hidden case), which is why this bug survives
  simple tests.
- **Stamping a partly matched node.** Setting the time when the walk enters an
  edge, before measuring the match, refreshes the lower half of a split: tokens
  that the request did not reuse look recently used.
- **A split that drops the lock count.** The upper half starts at 0 while the
  request that locked the lower half still depends on it. In this problem only
  leaves are evicted, so the damage shows on `unlock`, which decrements the upper
  half to -1; in an engine that tracks evictable memory by lock count, the
  accounting is off by the upper half's tokens.
- **Locking the node only.** The same accounting error in the other direction,
  and after a split the inherited count on the upper half is never released: the
  branch can never be evicted.
- **No cascade.** If a parent that just became a leaf is not reconsidered,
  `evict` returns less than was asked although more could go, and the caller,
  which needed the memory for a request, fails or retries.
- **Ignoring locks**, **evicting the newest leaf**, or **continuing after
  `n_tokens` is reached** (`<=` for `<`): each is one comparison.

## Complexity

With $n$ the length of the input, $d$ the number of nodes on its path, and $N$
the number of nodes in the tree: the walk is $O(n)$ token comparisons plus $O(d)$
dict lookups; a split is $O(\text{edge length})$ for the slices; `lock` and
`unlock` are $O(d)$; `evict` is $O(N)$ to collect the leaves plus $O(\log N)$ per
removal. Memory is $O(\text{total tokens})$. An engine keeps the set of evictable
leaves up to date instead of collecting it on every call.

## In the wild

- **The paper.** Zheng et al., "SGLang: Efficient Execution of Structured Language
  Model Programs" (NeurIPS 2024), introduces RadixAttention: the KV cache of
  finished requests is kept in a radix tree keyed by tokens, with LRU eviction of
  leaves and a reference counter on nodes used by running requests.
- **The code** (`python/sglang/srt/mem_cache/radix_cache.py`). `match_prefix`
  walks the tree and, when the match ends inside a node's key, calls
  `_split_node`, which creates the new *upper* node, sets
  `new_node.lock_ref = child.lock_ref`, and keeps `child` as the lower part, as
  here. `inc_lock_ref` and `dec_lock_ref` loop `while node != self.root_node`.
  `evict` heapifies the evictable leaves by priority (last access time under the
  default LRU policy), deletes the popped leaf, and pushes the parent when
  `len(x.parent.children) == 0 and x.parent.lock_ref == 0`.
- **Where this problem differs.** (1) SGLang stamps `child.last_access_time`
  when the walk enters a child, before it knows how much of the key matches, so
  both halves of a split carry the new time; with wall-clock times the lower half
  and a newly inserted sibling can then tie. This problem stamps only fully
  matched nodes, which is the stricter reading of "used" and makes the eviction
  order unique. (2) Time is `time.monotonic()` there and a logical counter here.
  (3) SGLang matches and splits at multiples of a page size when the allocator is
  paged, and stores the KV indices of the tokens as each node's value. (4) Other
  eviction policies (LFU, priority) plug into the same heap.
- **Compared with block hashing.** vLLM's index (`block-hash-prefix-cache`) is a
  flat hash table with $O(1)$ probes per block and hits in multiples of the block
  size; the radix tree matches to the token and shares structure explicitly, at
  the cost of pointer-chasing and splits. Both evict from the tail of a prefix
  inward, for the same reason.
