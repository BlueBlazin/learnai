## The idea

Because a block table entry is just a physical block id, multiple block tables can point to the same physical block. Forking a sequence only copies a list of integers rather than gigabytes of keys and values. Two rules keep sharing safe.

**Rule 1: reference counts decide when a block is free.** Each block's reference count is the number of live block tables containing it. `fork` increments the count for every block in the parent's table, `free` decrements each block's count, and a block returns to the free pool only when its count reaches `0`. No sequence owns a block exclusively. A parent can be freed before its children, and its blocks stay alive as long as any child references them.

**Rule 2: only a block's sole holder may write to it.** KV blocks are append-only, so once a slot is written it never changes, and a full block is never written again. Full blocks stay shared indefinitely. The only write that can ever hit a shared block is an append into a *partly filled last block*. Before writing, the appending sequence checks that block's reference count. If the count is `1`, it writes in place. If the count is greater than `1`, it allocates a fresh block, replaces the last entry in its own table, decrements the old block's reference count, and returns `(old, new)` so the engine copies the old block's contents before writing. Other holders keep pointing to the untouched old block.

Each fork triggers at most one block copy per sequence, copying at most `block_size - 1` valid slots. That small bound is what makes copy-on-write cheap in PagedAttention.

**Evaluate sharing at append time.** Suppose $m$ sequences share a partly filled block. The first $m - 1$ sequences to append each trigger a copy and decrement the original block's reference count. When the last sequence appends, the count is `1`, so it writes in place. There are $m - 1$ copies, not $m$. Freeing siblings has the same effect. Once only one holder remains, it can write in place again.

**Capacity check.** A copy needs a free block just like growth does. Keep `append` transactional by checking `copy + growth <= len(free)` before mutating any state.

## Reference solution (Python)

```python
import heapq


class OutOfBlocks(Exception):
    pass


class CowAllocator:
    def __init__(self, n_blocks, block_size):
        self.block_size = block_size
        self._free = list(range(n_blocks))
        self._refs = [0] * n_blocks
        self._tables = {}
        self._tokens = {}

    def append(self, seq_id, n_tokens):
        table = self._tables.get(seq_id, [])
        tokens = self._tokens.get(seq_id, 0)
        writes_last = tokens % self.block_size != 0
        copy = int(writes_last and self._refs[table[-1]] > 1)
        growth = -(-(tokens + n_tokens) // self.block_size) - len(table)
        if copy + growth > len(self._free):
            raise OutOfBlocks(f"need {copy + growth} blocks, {len(self._free)} free")
        new = [heapq.heappop(self._free) for _ in range(copy + growth)]
        for block in new:
            self._refs[block] = 1
        table = list(table)
        copies = []
        if copy:
            source, table[-1] = table[-1], new[0]
            self._refs[source] -= 1
            copies.append((source, new[0]))
        self._tables[seq_id] = table + new[copy:]
        self._tokens[seq_id] = tokens + n_tokens
        return new, copies

    def fork(self, parent_id, child_id):
        if child_id in self._tables:
            raise ValueError(f"sequence {child_id!r} is already live")
        table = self._tables[parent_id]
        for block in table:
            self._refs[block] += 1
        self._tables[child_id] = list(table)
        self._tokens[child_id] = self._tokens[parent_id]

    def free(self, seq_id):
        for block in self._tables.pop(seq_id):
            self._refs[block] -= 1
            if self._refs[block] == 0:
                heapq.heappush(self._free, block)
        del self._tokens[seq_id]

    def block_table(self, seq_id):
        return list(self._tables[seq_id])

    def ref_count(self, block):
        return self._refs[block]

    def free_blocks(self):
        return len(self._free)
```

In `append`, the source of a copy is never pushed back to `_free`. Its reference count was at least `2`, so after decrementing it, at least one other sequence still holds it.

## Common mistakes

- **Copying eagerly on fork.** Giving the child its own copy of every block avoids corruption, but defeats the purpose of sharing. Memory doubles on every fork, each fork copies the full prompt, and a fork can fail with `OutOfBlocks`.
- **Skipping the copy on write.** Writing into a shared last block lets two sequences write different tokens into the same slot. The second write overwrites the first, so the first sequence silently attends to another sequence's keys and values without raising an error.
- **Double free.** Returning every block in a freed sequence's table to the pool without checking reference counts hands live blocks to new requests while siblings still read them. This fails immediately when a parent finishes before its children, which is common in parallel sampling.
- **Marking blocks as shared at fork time.** Tracking a sticky "shared" flag set during `fork` forces the last remaining holder to copy as well, causing $m$ copies instead of $m - 1$ and leaking or delaying the release of the original block. Always check the current reference count at append time.
- **Copying a full last block.** When `tokens % block_size == 0`, the new tokens go into a newly allocated growth block and never touch the previous block. Copying it wastes a block per fork at every block boundary.
- **Forgetting to decrement the source's reference count.** After a copy, the appending sequence no longer holds `src`. Leaving `src`'s count unchanged prevents it from ever reaching `0`, permanently leaking the block.
- **Excluding the copy block from the capacity check.** Checking only growth blocks and popping the copy destination afterward fails halfway through when the pool is one block short, leaving mutated state behind.
- **Aliasing block tables.** Setting `tables[child] = tables[parent]` shares the same list object. Any in-place `append` or `extend` then mutates both sequences' tables.
- **Allocating the copy destination last.** Although engines do not care about allocation order, this problem requires allocating the copy destination before growth blocks (lowest id first) so outputs are deterministic.
- **Initializing a forked child with zero tokens.** `fork` must copy the parent's token count along with its block table, or the child's first append will write to slot `0` of the shared prefix.

## Complexity

Let $n$ be `n_blocks`, $k$ be the length of a sequence's block table, and $g$ be the number of newly allocated blocks (`copy + growth`). `fork` takes $O(k)$ time. `free` takes $O(k)$ time plus $O(\log n)$ for each block whose count drops to `0`. `append` takes $O(k + g \log n)$ time because it copies the table list. Production engines mutate the sequence's own list in place to avoid the $O(k)$ copy. Extra memory is $O(n)$ for the reference-count array.

## In the wild

- **The PagedAttention paper.** Kwon et al. (SOSP 2023, Section 4.4) introduce this reference-counted copy-on-write design for parallel sampling and beam search. Multiple sequences map logical blocks to the same physical block, and any block that more than one sequence would write to is copied at block granularity, mirroring OS process `fork`. Only the last, partly filled block of a shared prompt is ever copied.
- **vLLM's original block manager.** In `vllm/core/block_manager.py` (v0.2), `fork` copies the parent's block table and increments `ref_count` on every block, noting that it allocates no physical blocks and is therefore safe from OOM. `append_slot` returns `None` when the last block has `ref_count == 1`, and otherwise allocates a new block, replaces the last table entry, decrements the old block's count, and returns `(old_block, new_block)` for the engine to copy. Freeing a block whose count is already `0` raises `"Double free!"`.
- **Prefix caching today.** Modern vLLM shares blocks through a prefix cache instead (`block-hash-prefix-cache`). A new request with the same prompt looks up full blocks by hash and increments their reference counts. Because only full blocks are cached, no copy-on-write is needed, and the partly filled tail is simply recomputed.
