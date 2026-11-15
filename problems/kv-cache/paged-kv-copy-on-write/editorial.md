## The idea

In a paged cache a block table entry is just an id, so nothing stops two tables
from naming the same physical block. Sharing is then free: a fork copies a list of
integers, not gigabytes of keys and values. Two rules make sharing safe.

**Rule 1: reference counts decide when a block is free.** A block's count is the
number of tables that contain it. `fork` increments, `free` decrements, and a
block returns to the pool only at 0. No sequence "owns" a block in any stronger
sense: the parent may be freed first, and the blocks it allocated live on in its
children.

**Rule 2: a block is written only by its sole holder.** KV blocks are append-only:
a slot, once written, never changes, and a full block is never written again. So
full blocks can stay shared forever, and the only write that could ever hit a
shared block is an append into a *partly filled last block*. Before that write,
the appending sequence checks the count. If it is 1, it writes in place. If it is
more, it takes a fresh block, repoints its own table entry, drops one reference on
the old block, and tells the engine to copy old to new. The other holders keep the
old block untouched.

The copy is of one block, at most once per sequence per fork, and it copies at
most `block_size - 1` valid slots. That bound is why PagedAttention can afford it.

**"Shared" is evaluated at the append.** After a fork of two sequences, the first
to append copies. The second then holds the old block alone (count 1) and writes
in place. With $n$ sequences sharing a partly filled block, there are $n - 1$
copies, not $n$. A freed sibling has the same effect: its references disappear,
and the survivor may write in place again.

**Capacity.** The copy needs a block too. The transactional rule of the previous
problem still holds, so the check is `copy + growth <= free` before anything is
changed.

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

The source of a copy is never pushed back to the pool in `append`: its count was
above 1, so after the decrement at least one other sequence still holds it.

## Common mistakes

- **Copying on fork.** Giving the child its own copy of every block is correct in
  the sense that nothing is corrupted, and it throws away the point: memory
  doubles with every fork, each fork costs a full memcpy of the prompt, and a fork
  can fail for lack of memory. Beam search forks at every step.
- **No copy.** Appending into a shared last block makes two sequences write their
  different token 6 into the same slot. The second write wins, and the first
  sequence silently attends to another sequence's keys and values from then on.
  Nothing crashes; the text of one sample just goes subtly wrong.
- **Double free.** `free` that returns every block of the table to the pool
  ignores the other holders. The block is handed to the next request while a
  sibling still reads it. The classic trigger is the parent finishing before its
  children, which is the normal order for parallel sampling.
- **Deciding "shared" at fork time.** Marking blocks as shared when they are
  forked and copying whenever a marked block is written makes the *last* holder
  copy as well: $n$ copies instead of $n - 1$, and the abandoned original leaks or
  is freed late. Ask the count, now.
- **Copying a full block.** If the last block is full, the append goes to a new
  block and writes nothing old. Copying it anyway wastes a block per fork at
  every block boundary.
- **Forgetting to drop the source's reference.** After the copy this sequence no
  longer holds the old block. If its count stays where it was, the block can never
  reach 0 and is lost to the pool.
- **Leaving the copy out of the capacity check.** Checking only the growth blocks
  and taking the copy destination afterwards fails half-way when the pool is one
  block short: the sequence has grown, and then the call raises.
- **Aliased tables.** `tables[child] = tables[parent]` stores one list under two
  keys. Combined with in-place `append`/`extend` on tables, every block one
  sequence gains shows up in the other.
- **The copy destination allocated last.** Harmless in an engine, but this problem
  fixes the order (copy first, lowest id first) so that results are reproducible.
- **A child that starts at zero tokens.** The fork must copy the token count along
  with the table; otherwise the child's first append is placed at slot 0 of the
  shared prefix.

## Complexity

`fork` and `free` are $O(k)$ for a table of $k$ blocks (plus $O(\log n)$ per block
that returns to the heap); `append` is $O(k)$ for the table copy plus
$O(g \log n)$ for $g$ new blocks. Engines avoid the table copy by mutating the
table in place, which is safe once every sequence has its own list. Memory is one
counter per block.

## In the wild

- **The paper.** PagedAttention (Kwon et al., SOSP 2023, section 4.4, on other
  decoding scenarios) describes this design for parallel sampling and beam
  search: several sequences map their logical blocks to the same physical block,
  each physical block carries a reference count, and a block that more than one
  sequence would modify is copied at block granularity, as an operating system
  does for the pages of a forked process. Only the last, partly filled block of a
  shared prompt is ever copied.
- **vLLM's first block manager** (`vllm/core/block_manager.py`, v0.2): `fork`
  copies the parent's block table and increments `ref_count` on every block, with
  the comment that it "does not allocate a new physical block" and is therefore
  "always safe from OOM". `append_slot` returns `None` when the last block has
  `ref_count == 1` and otherwise allocates a block, replaces the last table entry,
  frees the old block (a decrement), and returns the pair
  `(old block number, new block number)` for the engine to copy. The allocator's
  `free` raises `"Double free!"` on a block whose count is already 0.
- **Today.** vLLM's current KV cache manager shares blocks through the prefix
  cache instead: a second request with the same prompt finds the full blocks by
  hash and takes a reference (`block-hash-prefix-cache`). Only full blocks are
  shared there, so no copy on write is needed: the partly filled tail is simply
  recomputed.
