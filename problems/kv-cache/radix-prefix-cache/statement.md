Implement the index of a token-level prefix cache as a radix tree: every cached
sequence is a path from the root, sequences that start alike share the nodes of
their common prefix, and each edge carries a run of tokens rather than one. Lookup
is a walk; the work is in keeping the tree exact while edges are split, paths are
locked by running requests, and old leaves are evicted.

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

**The tree.** The root stands for the empty prefix. Every other node has an
**edge** from its parent that carries one or more tokens, and stands for the
**prefix** made of all tokens on the path from the root through its edge. No two
children of a node have edges that start with the same token, so a token sequence
follows at most one path. Each node other than the root has a **lock count**
(initially 0) and a **last access time**. The cache stores no tensors; in an
engine, each node would own the KV entries of the tokens on its edge.

**The clock.** The cache counts accesses: the clock starts at 0, and every call of
`match_prefix` or `insert` first advances it by 1. No other method advances it.

**The walk**, shared by `match_prefix` and `insert`. Starting at the root, follow
`tokens` down the tree for as long as they agree with it. The **matched length**
$L$ is the length of the longest prefix of `tokens` that is a prefix of some node's
prefix. If the walk ends inside an edge (the first $c \ge 1$ tokens of the edge
matched, and the edge has more), that edge is **split** after $c$ tokens:

- a new *upper* node takes the first $c$ tokens and becomes the child of the old
  parent; the existing node keeps the remaining tokens, its children, its lock
  count, its access time, and its identity, and becomes the upper node's only
  child;
- the upper node **inherits** the lock count and the access time of the existing
  node.

After the walk, a node ends exactly at length $L$ (the root if $L = 0$). Finally
every node on the path from the root to that node, the upper half of a split
included, gets the current clock value as its last access time. A node whose edge
was only partly matched (the lower half of a split) is not on that path and keeps
its time.

- `match_prefix(tokens)` performs the walk and returns `(L, node)`, where `node` is
  a **handle** to the node that ends at length $L$: any object of your choice, to
  be passed to `lock` and `unlock`. A handle keeps denoting the same prefix when
  edges above or below it are split later, and it is valid until its node is
  evicted.
- `insert(tokens)` performs the walk and then, if $L <$ `len(tokens)`, adds one new
  leaf under the node that ends at $L$, with the edge `tokens[L:]`, lock count 0,
  and the current clock value. Returns $L$, the number of tokens that were already
  in the tree. (When `tokens` is already a prefix in the tree, the walk may split
  an edge, and nothing is added.)
- `lock(node)` adds 1 to the lock count of the node and of every ancestor except
  the root. `unlock(node)` subtracts 1 from the same nodes. On the root's handle
  both do nothing. Both return `None`. The tests unlock a handle only as often as
  they locked it.
- `evict(n_tokens)` removes nodes until at least `n_tokens` tokens have been
  removed or nothing can be removed. It repeats: among the **leaves** (nodes
  without children, the root excepted) whose lock count is 0, remove the one with
  the smallest last access time, and count the tokens of its edge. A node whose
  last child was just removed is a leaf from then on and takes part in the same
  call. Inner nodes are never removed, and no merging of edges takes place.
  Returns the number of tokens removed (which is 0 for `n_tokens = 0`, and may
  exceed `n_tokens`, since leaves are removed whole).
- `total_tokens()` returns the number of tokens on all edges.
- `nodes()` returns one tuple `(prefix, edge_length, locks)` for every node except
  the root, sorted in ascending order: `prefix` is the node's prefix as a tuple of
  ints, `edge_length` the number of tokens on its own edge, `locks` its lock
  count.

`tokens` is a list of ints and may be empty; do not modify it.

> [!NOTE]
> Two leaves never have the same last access time: one access stamps nodes on a
> single path, and a path contains at most one leaf. So the order of eviction is
> fully determined.

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

The second insert shares three tokens with the first sequence, so the edge
`1 2 3 4 5 6` is split into `1 2 3` and `4 5 6`, and `7 8` becomes a second child.
The match of `1 2 3 4 9` ends one token into the edge `4 5 6`, so that edge is
split as well, and the handle `h` denotes the node `[4]`, whose prefix is
`1 2 3 4`. The lower half `[5 6]` was not matched and keeps time 1. Locking `h`
locks `[4]` and `[1 2 3]`.

```python
cache.evict(4)  # 4    removes [5 6] (time 1), then [7 8] (time 2)
cache.evict(100)  # 3    removes [9 9 9]; [4] is a leaf now, but locked
cache.unlock(h)
cache.evict(1)  # 1    removes [4]
cache.total_tokens()  # 3    only [1 2 3] is left
cache.evict(1)  # 3    a leaf goes whole
```

## Constraints

- Sequences of up to a few hundred tokens, at most a few hundred calls per cache,
  token ids are non-negative ints.
- Everything is compared exactly, after every operation: the value the operation
  returns, the full list `nodes()` (so every split, every lock count, and every
  removal is checked), and `total_tokens()`.
- **Property.** On random workloads the result is compared with a brute-force
  model that keeps only the set of node prefixes: the matched length must equal
  the longest common prefix of `tokens` with any prefix in the set.
- Handles are used as the specification allows: locked or unlocked only while
  their node exists.

## Why this matters

A hash of fixed blocks finds a shared prefix only up to the last block boundary
and only where the grid lines up. A radix tree matches to the token: a chat that
grows by one turn, an agent that retries with a different tool result, few-shot
prompts that share their examples and differ in the question, all reuse exactly
what they have in common. This is the structure behind SGLang's RadixAttention.
It lives in the scheduler's hot path, and its three invariants map to three
failure modes. An edge that is not split where a match ends gives the new request
either too little (a cache miss on half a prompt) or a node whose KV it only
partly owns. A lock count that is wrong lets the evictor free the prefix a running
request is attending to. And an evictor that removes an inner node, or the wrong
leaf, destroys the shared system prompt while keeping the one-off suffixes that
hang below it.
