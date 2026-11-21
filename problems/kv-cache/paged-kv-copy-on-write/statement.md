Extend the paged KV allocator so that sequences can share blocks: `fork` creates a
sequence whose block table points at its parent's blocks, and a block is copied only
when a sequence is about to write into one that another sequence also holds. The
bookkeeping is a reference count per block; the difficulty is that "shared" is a
fact about the present, and it changes whenever any sequence copies, appends, or is
freed.

![This problem is the Block pool, here with blocks that several sequences share.](assets/map.svg)

## Specification

```python
class OutOfBlocks(Exception): ...


class CowAllocator:
    def __init__(self, n_blocks, block_size): ...
    def append(self, seq_id, n_tokens): ...  # -> (allocated, copies)
    def fork(self, parent_id, child_id): ...  # -> None
    def free(self, seq_id): ...  # -> None
    def block_table(self, seq_id): ...  # -> list of block ids
    def ref_count(self, block): ...  # -> int
    def free_blocks(self): ...  # -> int
```

The pool, the block tables, and the placement of tokens are those of
`paged-kv-allocator`: `n_blocks` physical blocks with ids `0` to `n_blocks - 1`,
each with room for `block_size` tokens; a live sequence with $t \ge 1$ tokens has a
block table of exactly $\lceil t / \texttt{block\_size} \rceil$ ids, and its token
$i$ lives in slot `i % block_size` of block `table[i // block_size]`. New here: one
block id may appear in the tables of several sequences.

The **reference count** of a block is the number of live sequences whose block
table contains it. A block is free exactly when its count is 0.

- `fork(parent_id, child_id)` creates the sequence `child_id` with the same token
  count and the same block table as `parent_id` (the same ids, in a list of its
  own). Every block of the parent gains one reference. No block is allocated, so a
  fork never raises `OutOfBlocks`. Raises `ValueError` if `child_id` is live
  (this includes `child_id == parent_id`), and otherwise `KeyError` if `parent_id`
  is not live; in both cases nothing changes. After the fork the two sequences are
  equals: either may append, fork again, or be freed first.

- `append(seq_id, n_tokens)` adds `n_tokens` $\ge 1$ tokens; a `seq_id` that is not
  live starts a new sequence. With $t$ tokens before the call:

  1. **Copy on write.** The call writes into the sequence's last block if that
     block has empty slots ($t \bmod \texttt{block\_size} \ne 0$). If that block's
     reference count is then greater than 1, the sequence must not write into it:
     a new block `dst` is allocated, it replaces the last entry `src` of **this
     sequence's** table, and `src` loses one reference. The engine will copy the
     contents of `src` to `dst` before it writes the new tokens. A full last block
     is never written again, so it is never copied, however many sequences hold
     it.
  2. **Growth.** As before,
     $\lceil (t + n) / \texttt{block\_size} \rceil - \lceil t / \texttt{block\_size} \rceil$
     further blocks are allocated and appended to the table.

  Blocks are taken from the free blocks lowest id first: the copy destination (if
  there is one) first, then the growth blocks. Returns a tuple
  `(allocated, copies)`:

  - `allocated`: the list of all block ids allocated by this call, in allocation
    order (so `dst` is `allocated[0]` when there is a copy);
  - `copies`: `[(src, dst)]` if the call copied, otherwise `[]`. Each pair is a
    tuple of two ints.

  If the call needs more blocks than are free, counting the copy, it raises
  `OutOfBlocks` and **changes nothing**.

- `free(seq_id)` forgets the sequence. Each block of its table loses one reference,
  and a block whose count reaches 0 returns to the pool. Returns `None`. Raises
  `KeyError` if `seq_id` is not live.
- `block_table(seq_id)` returns the block table as a new list. Raises `KeyError` if
  `seq_id` is not live.
- `ref_count(block)` returns the reference count of the block with id `block`
  ($0 \le$ `block` $<$ `n_blocks`), 0 for a free block.
- `free_blocks()` returns the number of blocks with reference count 0.

`seq_id` is any hashable value.

> [!NOTE]
> Whether an append copies depends on the reference count **at the time of the
> append**. After a fork, the first of the two sequences to append copies the
> shared last block; the other is then its only holder and writes in place.

## Example

```python
alloc = CowAllocator(n_blocks=6, block_size=4)

alloc.append("a", 6)  # ([0, 1], [])        a: 6 tokens; block 1 holds 2 of 4
alloc.fork("a", "b")  #                     b: table [0, 1]; ref counts 2 and 2
alloc.append("b", 1)  # ([2], [(1, 2)])     block 1 is shared and partly filled:
#                     b moves to a copy. b: [0, 2]
alloc.ref_count(1)  # 1                   only a holds block 1 now
alloc.append("a", 1)  # ([], [])            so a writes in place: no copy
alloc.append("b", 4)  # ([3], [])           b: 11 tokens, table [0, 2, 3]
alloc.fork("b", "c")  #                     c: table [0, 2, 3]; ref_count(0) is 3
alloc.free("b")  #                     nothing returns to the pool: c holds
alloc.free_blocks()  # 2                   every block b held (blocks 4, 5 free)
alloc.append("c", 2)  # ([4], [])           block 3 (3 of 4) is c's alone: in place,
#                     then one new block. c: [0, 2, 3, 4]
alloc.free("a")  #                     block 1 returns; block 0 is still c's
alloc.free_blocks()  # 2                   (blocks 1 and 5)
alloc.free("c")
alloc.free_blocks()  # 6
```

The third call is the copy on write: `a` and `b` agree on tokens 0 to 5, and token
6 is where they differ. `b` cannot put its token 6 into slot 2 of block 1, because
`a` will put its own token 6 there. So `b` gets block 2, the engine copies block 1
into it (two valid slots), and `b` writes its token into slot 2 of the copy. Block
0 is full; it stays shared for as long as either sequence lives.

`free("b")` shows why counts, not ownership, decide what returns to the pool: `b`
allocated blocks 2 and 3, but `c` still reads them.

![The pool at four points of the example, with each block's reference count and holders.](assets/pool-steps.svg)

## Constraints

- $1 \le$ `n_blocks` $\le 1000$, $1 \le$ `block_size` $\le 64$, and at most a few
  hundred calls per allocator.
- Results are compared exactly: the tuple returned by `append`, every block table,
  every reference count, and `free_blocks()` after every operation.
- **Checks after every operation**, on scripted and random workloads:
  - `ref_count(b)` equals the number of block tables that contain `b`, and
    `free_blocks()` plus the number of distinct blocks in block tables equals
    `n_blocks`.
  - *Isolation.* The tests play the engine: they keep the contents of every
    physical block, perform the copies `append` returns, write each new token to
    the slot its sequence's block table names, and then read every live sequence
    back through its table. Each sequence must read exactly the tokens it holds:
    its ancestors' up to the fork, then its own.
- **Copies and allocations are counted.** Four sequences that share a 10-token
  prompt (`block_size = 4`: two full blocks and one half-filled) and then decode 7
  tokens each must cause at most **3** copies and **14** block allocations: 3 for
  the prompt, one copy for each of the first three sequences to decode (the fourth
  is by then the only holder of the prompt's last block), and 2 growth blocks per
  sequence. Copying eagerly on fork would take 20.
- `OutOfBlocks` must be an exception class defined by your solution, as in the
  starter.

## Why this matters

One prompt often feeds several sequences: `n > 1` completions of the same request,
the beams of a beam search, the branches of a tree search or of speculative
tool calls. All of them share the prompt's keys and values, which are usually most
of the memory. With reference-counted blocks the shared part is stored once, a
fork costs no memory and no copying, and the only copy ever made is of one block:
the partly filled one at the point where two sequences diverge. This is how
PagedAttention makes parallel sampling and beam search cheap, and the same
mechanism lets a prefix cache hand a cached system prompt to many requests. Its
bugs are the bugs of every shared-ownership scheme: a missed copy lets one request
write into another's attention state, a block freed while a sibling still reads it
is handed to a stranger, and a count that is never decremented leaks capacity
until restart.
