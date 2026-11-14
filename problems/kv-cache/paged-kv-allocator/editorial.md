## The idea

Paging separates *where a sequence thinks its tokens are* from *where they are*.
Logically, a sequence's cache is one contiguous array: token $i$ is in logical block
$i\ //\ B$, slot $i \bmod B$, with $B$ the block size. Physically, each logical block
is whichever free block the pool had when it was needed. The block table is the map
between the two, and the attention kernel follows it to gather keys and values.

Because blocks are filled left to right and only the last one can be partly empty,
the number of blocks a sequence owns is a function of its token count alone,
$\lceil t / B \rceil$. That makes the allocation rule a subtraction:

$$
\text{new blocks} = \left\lceil \frac{t + n}{B} \right\rceil - \left\lceil \frac{t}{B} \right\rceil ,
$$

and there is no need to track "slots left in the last block" as separate state that
could drift out of sync.

The second idea is transactional: decide whether the whole append can succeed
*before* modifying anything. Then a refusal has nothing to undo.

## Reference solution (Python)

```python
import heapq


class OutOfBlocks(Exception):
    pass


class BlockAllocator:
    def __init__(self, n_blocks, block_size):
        self.block_size = block_size
        self._free = list(range(n_blocks))      # a sorted list is a valid min-heap
        self._tables = {}
        self._tokens = {}

    def append(self, seq_id, n_tokens):
        table = self._tables.get(seq_id, [])
        total = self._tokens.get(seq_id, 0) + n_tokens
        needed = -(-total // self.block_size) - len(table)
        if needed > len(self._free):
            raise OutOfBlocks(f"need {needed} blocks, {len(self._free)} free")
        new = [heapq.heappop(self._free) for _ in range(needed)]
        self._tables[seq_id] = table + new
        self._tokens[seq_id] = total
        return new

    def free(self, seq_id):
        for block in self._tables.pop(seq_id):
            heapq.heappush(self._free, block)
        del self._tokens[seq_id]

    def block_table(self, seq_id):
        return list(self._tables[seq_id])

    def free_blocks(self):
        return len(self._free)
```

`self._tables.get(seq_id, [])` reads without creating an entry, so a new sequence
comes into existence only on the two commit lines. `dict.pop` and indexing raise
`KeyError` for a sequence that is not live, which is the specified behavior.

## Common mistakes

- **A block per token.** Allocating `n_tokens` blocks treats the block as a slot. It
  is the right answer only for `block_size = 1`, and it exhausts the pool
  `block_size` times too early.
- **Rounding each append on its own** (`ceil(n_tokens / block_size)` new blocks per
  call). Decoding appends one token at a time, so this is again a block per token,
  and the block table no longer matches the positions of the tokens.
- **`total // block_size + 1` blocks.** Right except when the count is a multiple of
  the block size, where it allocates a block that holds nothing. A sequence that
  ends exactly at a block boundary then keeps one block too many, and a pool that
  should fit the request refuses it.
- **Partial allocation.** Popping blocks one by one and raising when the pool runs
  dry leaves the blocks popped so far attached to the sequence (or to nothing). The
  caller sees an exception and assumes nothing happened; the memory is gone. Check
  first, then allocate.
- **A ghost sequence.** `setdefault(seq_id, [])` at the top of `append` registers the
  sequence before the check, so a refused first append leaves a live sequence with
  zero tokens.
- **`free` that does not restore the pool.** Deleting the table is not enough: the
  ids must go back to the free set, or capacity shrinks with every finished request.
- **A stack as the free list.** Appending freed ids to a list and popping from its
  end is a perfectly good allocator, but it is not "lowest id first": after
  `free("c"); free("a")` the next block would be one of `a`'s old ones in reverse
  order. This problem fixes the order so that results are reproducible.
- **Returning the internal list** from `block_table`. The caller's edits then
  rewrite the allocator's state.

## Complexity

`append` is $O(k \log n)$ for $k$ new blocks out of $n$, `free` is $O(k \log n)$ for
a sequence of $k$ blocks, and the queries are $O(1)$ plus the copy. Memory is $O(n)$
ids plus one table entry per owned block. With a plain stack instead of a heap every
operation is $O(k)$, which is what engines use, since they do not care which free
block they get.

## In the wild

- **The paper.** In vLLM (Kwon et al., SOSP 2023) a KV block holds the keys and
  values of a fixed number of tokens, the block table records for each logical block
  its physical block and how many of its slots are filled, and "a new physical block
  is only allocated when all previous blocks are full", which limits the waste of a
  request to one block. The paper's default block size is 16 tokens: large enough for
  the GPU to read a block efficiently, small enough to keep that waste low. Its
  Figure 6 is the example of the second public test: a 7-token prompt occupies two
  blocks of four, the next token fills the last slot, and the one after it gets a
  third block somewhere else in the pool.
- **Which block.** Real allocators do not hand out the lowest id. Today's vLLM keeps
  free blocks in a doubly linked queue and takes from its front; freed blocks that
  hold a cached prefix go to the back, so that they are reused last and can still be
  found by a later request with the same prefix (`block-hash-prefix-cache`).
- **Sharing.** Because a table entry is just an id, two sequences can point at the
  same physical block. With a reference count per block and copy-on-write for a
  shared last block, parallel samples and beams share their common prefix
  (`paged-kv-copy-on-write`).
- **Running out.** `OutOfBlocks` is where scheduling begins: vLLM preempts the most
  recently arrived sequences, frees all of their blocks (all or nothing per
  sequence), and later recomputes or swaps their cache back in
  (`preemption-recompute`).
