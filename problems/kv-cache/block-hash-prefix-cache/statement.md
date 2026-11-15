Build the index behind automatic prefix caching: token sequences are cut into
fixed-size blocks, every full block gets a key, and a new request reuses the KV
blocks of the longest prefix whose keys are already in the cache. The key is what
makes it correct: the keys and values in a block depend on every token before the
block, so a block's key must identify its whole prefix, not just its own tokens.

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

The cache stores no tensors. It decides which blocks exist, who is using them, and
which one goes when room is needed.

**Blocks.** With $B$ = `block_size`, block $k$ of a token list `tokens` is
`tokens[k*B : (k+1)*B]`. Only **full** blocks count: a list of $n$ tokens has
$\lfloor n / B \rfloor$ blocks, and the $n \bmod B$ tokens after the last full block
are ignored by every method.

**Identity.** Block $k$ of one list and block $k'$ of another are the same cached
block exactly when $k = k'$ and the lists agree on **all** tokens up to the end of
the block, `tokens[: (k+1)*B]`. The intended implementation gives every block a
key computed from its parent's key and its own tokens,

$$
\text{key}_k = H(\text{key}_{k-1},\ \text{tokens}[kB : (k+1)B]), \qquad \text{key}_{-1} = \text{a constant},
$$

so that a key stands for the whole prefix while costing $O(B)$ to compute. Any
scheme with the identity above is accepted, provided two different prefixes never
share a key in practice (a 64-bit or wider hash, or exact keys).

**State.** The cache holds at most `capacity` blocks. Each cached block has a
**reference count**: the number of active requests that hold it. The cached blocks
with count 0 form the **eviction order**, a queue: a block joins at the back when
its count drops to 0 and leaves when it is referenced again or evicted.

- `lookup(tokens)` returns the number of **leading** blocks of `tokens` that are
  cached: the largest $h$ such that blocks $0, \ldots, h-1$ are all in the cache.
  It changes nothing: no count, no position in the eviction order.

- `acquire(request_id, tokens)` starts a request with this prompt and returns
  `(hits, held)`:

  1. `hits = lookup(tokens)`. Each of these blocks gains one reference; a block
     whose count was 0 leaves the eviction order.
  2. Then, for each further block of `tokens` in order: if the cache holds
     `capacity` blocks, the block at the **front** of the eviction order is evicted
     (removed from the cache); if the eviction order is empty, stop, and this block
     and all later ones are not cached. Otherwise the block is inserted with
     reference count 1.
  3. `held` is `hits` plus the number of blocks inserted. The request holds blocks
     $0, \ldots, \text{held} - 1$ of `tokens` until it is released.

  Step 1 is completed before step 2 begins. Raises `ValueError`, changing nothing,
  if `request_id` is already active.

- `release(request_id)` ends a request. Its blocks lose one reference each,
  **deepest block first** (block `held - 1`, then `held - 2`, down to block 0), and
  a block whose count reaches 0 joins the back of the eviction order at that
  moment. Returns `None`. Raises `KeyError` if `request_id` is not active.

- `ref_count(tokens)` returns the reference count of the **last** block of `tokens`
  (block $\lfloor n / B \rfloor - 1$). Raises `KeyError` if `tokens` has no full
  block or that block is not cached.

- `cached_blocks()` returns the number of blocks in the cache.

`request_id` is any hashable value. `tokens` is a list of ints and may be empty; do
not modify it.

> [!NOTE]
> Releasing deepest first puts a block ahead of its parent in the eviction order.
> Together with step 1 before step 2, this keeps the cache *prefix-closed*: a
> cached block's parent is always cached too, so no slot is ever spent on a block
> that `lookup` can no longer reach.

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

The third call is the reason for the parent key. The list
`[5, 6, 7, 8, 1, 2, 3, 4]` consists of two blocks whose tokens are both in the
cache, but its block 0 is not S's block 0, and its block 1 follows a different
prefix than S's block 0 did. Neither is a hit.

`release("r2")` queues `Q2`'s third block first and the first block of `S` last.
When `r4` needs three slots, the cache gives up the most specific blocks first and
the widely shared head of the system prompt last.

## Constraints

- $1 \le$ `block_size` $\le 64$, $1 \le$ `capacity` $\le 4096$, prompts of up to
  32,000 tokens, token ids are ints in $[0, 2^{31})$, and at most a few hundred
  calls per cache.
- Everything is compared exactly. After every operation the tests check its return
  value and then probe the cache with the read-only methods: `lookup` of every
  token list used so far, `ref_count` of every block that was ever cached
  (`KeyError` for those no longer cached), and `cached_blocks()`. Probing must not
  change the cache.
- Keying and looking up a prompt of 2,000 blocks must take well under a second.

## Why this matters

Chat requests repeat themselves: the same system prompt, the same tool
definitions, the same conversation with one more turn. With this index a server
skips the prefill of everything it has seen before, and nobody has to declare what
is shared, hence *automatic* prefix caching. vLLM's implementation is this
structure: a hash per full block chained to its parent, a reference count per
block, and a free queue from whose front blocks are evicted. Every rule above is
there because breaking it is a production incident. A key without the parent
serves one conversation's keys and values to another that happens to contain the
same sixteen tokens. A cached partial block is completed by one request and read,
stale, by the next. Evicting a referenced block frees memory a running request is
attending to. And a hash that can collide is a cross-tenant data leak, which is
why vLLM's default block hash has been SHA-256 since v0.11.
