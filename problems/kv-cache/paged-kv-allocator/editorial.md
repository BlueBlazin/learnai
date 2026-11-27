## The idea

Paging separates a sequence's logical layout from its physical storage. Let $B = \texttt{block\_size}$. Logically, a sequence's KV cache is one contiguous array where token $i$ sits in logical block `i // B` at slot `i % B`. Physically, each logical block can live in any free block in the pool. The block table maps logical block indices to physical block ids, and the attention kernel follows that table to gather keys and values.

Because slots fill from left to right and only the last block can be partly empty, a sequence of $t$ tokens always owns exactly $\lceil t / B \rceil$ blocks. When appending $n = \texttt{n\_tokens}$ tokens, the number of new blocks needed is

$$
\text{new blocks} = \left\lceil \frac{t + n}{B} \right\rceil - \left\lceil \frac{t}{B} \right\rceil .
$$

Computing the count directly from $t$ avoids tracking "slots remaining in the last block" as separate state that could drift out of sync.

The second idea is to make `append` transactional. Check whether enough free blocks exist *before* modifying the free pool, block tables, or token counts. When `OutOfBlocks` is raised, nothing needs to be rolled back.

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

`self._tables.get(seq_id, [])` reads without inserting a key, so a new sequence is registered only after the capacity check passes. `dict.pop` and dictionary indexing raise `KeyError` when `seq_id` is not live, matching the specification.

## Common mistakes

- **Allocating one block per token.** Allocating `n_tokens` blocks treats each block as a single slot. It works only when `block_size == 1` and exhausts the pool `block_size` times too early.
- **Rounding each append independently.** Computing `ceil(n_tokens / block_size)` per call ignores empty slots in the last block. During token-by-token decoding, this again allocates one block per token and breaks the mapping from token positions to slots.
- **Using `total // block_size + 1` instead of ceiling division.** This is off by one whenever `total` is an exact multiple of `block_size`, allocating an extra empty block and causing premature `OutOfBlocks` errors.
- **Partial allocation before checking capacity.** Popping blocks one by one until the pool runs dry leaves already-popped blocks leaked or attached to the sequence when `OutOfBlocks` is raised. Always check capacity before mutating state.
- **Creating a ghost sequence on a refused append.** Calling `setdefault(seq_id, [])` before the capacity check registers `seq_id` as a live sequence with zero tokens even when the first `append` fails.
- **Forgetting to return blocks in `free`.** Removing the sequence from `_tables` without pushing its block ids back into the free pool leaks capacity on every finished request.
- **Using a stack for the free list.** Appending freed ids to a list and popping from the end does not allocate the lowest id first. After freeing `"c"` and then `"a"`, the next allocation would return `"a"`'s blocks in reverse order instead of the lowest available id.
- **Returning the internal list from `block_table`.** If the caller mutates the returned list, it corrupts the allocator's internal state. Always return a copy.

## Complexity

Let $n$ be `n_blocks` and $k$ be the number of blocks allocated or freed in a call. `append` and `free` each take $O(k \log n)$ time with a min-heap. `free_blocks` takes $O(1)$ time, and `block_table` takes $O(L)$ time to copy a table of length $L$. Memory is $O(n)$ for the free heap and block table entries. Real serving engines typically use an $O(1)$-per-block stack or deque because they do not need lowest-id ordering.

## In the wild

- **The PagedAttention paper.** In vLLM (Kwon et al., SOSP 2023), each KV block holds a fixed number of tokens (16 by default, large enough for efficient GPU reads and small enough to bound internal fragmentation). A new physical block is allocated only when all previous blocks are full. Figure 6 of the paper matches the second public test, where a 7-token prompt fills two 4-token blocks, the 8th token fills the remaining slot, and the 9th token allocates a third block elsewhere in the pool.
- **Free block ordering.** Production allocators do not allocate the lowest id. Modern vLLM keeps free blocks in a doubly linked queue and pops from the front. Freed blocks that still hold cached prefixes go to the back so they are evicted last and can be reused by later requests (`block-hash-prefix-cache`).
- **Block sharing.** Because block table entries are just integer ids, multiple sequences can point to the same physical block. Adding per-block reference counts and copy-on-write lets parallel samples and beam search share a common prompt (`paged-kv-copy-on-write`).
- **Preemption.** Running out of blocks triggers scheduling decisions. When memory is full, vLLM preempts the most recently arrived sequences, frees all of their blocks at once, and later recomputes or swaps their KV cache back in (`preemption-recompute`).
