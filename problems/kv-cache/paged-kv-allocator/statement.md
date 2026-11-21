Write the bookkeeping of a paged KV cache: a pool of fixed-size physical blocks, and
for every sequence a *block table* that lists the blocks holding its keys and
values. A sequence gets a new block only when the ones it has are full, so no
sequence ever wastes more than one partly filled block.

![This problem is the Block pool, which hands fixed-size blocks to the sequences.](assets/map.svg)

## Specification

```python
class OutOfBlocks(Exception): ...

class BlockAllocator:
    def __init__(self, n_blocks, block_size): ...
    def append(self, seq_id, n_tokens): ...      # -> list of newly allocated block ids
    def free(self, seq_id): ...                  # -> None
    def block_table(self, seq_id): ...           # -> list of block ids
    def free_blocks(self): ...                   # -> int
```

The pool has `n_blocks` physical blocks with ids `0` to `n_blocks - 1`. Each block
has room for the keys and values of `block_size` tokens. Initially every block is
free and no sequence exists. The allocator stores no tensors; it only decides which
block belongs to whom.

A **live sequence** has a token count $t \ge 1$ and a block table: the ids of the
blocks it owns, in the order it received them. Token $i$ of the sequence (counting
from 0) lives in slot `i % block_size` of the block `table[i // block_size]`, so a
live sequence always owns exactly $\lceil t / \texttt{block\_size} \rceil$ blocks, and
only the last of them can have empty slots.

- `append(seq_id, n_tokens)` adds `n_tokens` tokens to the sequence `seq_id`. A
  `seq_id` that is not live starts a new sequence with zero tokens. The tokens first
  fill the empty slots of the sequence's last block; only what does not fit there
  gets new blocks. With $t$ tokens before the call, the number of new blocks is

  $$
  \left\lceil \frac{t + n}{\texttt{block\_size}} \right\rceil - \left\lceil \frac{t}{\texttt{block\_size}} \right\rceil .
  $$

  New blocks are taken from the free blocks **lowest id first** and appended to the
  block table in that order. Returns the list of newly allocated block ids, in
  allocation order; the list is empty if everything fit in the last block.

  If the call needs more blocks than are free, it raises `OutOfBlocks` and **changes
  nothing**: no block is taken, the token count and block table of the sequence stay
  as they were, and a `seq_id` that was not live is still not live.
- `free(seq_id)` returns all blocks of the sequence to the pool and forgets the
  sequence: `seq_id` is no longer live, and may be used again later for a new
  sequence. Returns `None`. Raises `KeyError` if `seq_id` is not live.
- `block_table(seq_id)` returns the block table as a new list (changing the returned
  list must not affect the allocator). Raises `KeyError` if `seq_id` is not live.
- `free_blocks()` returns the number of blocks that no sequence owns.

`seq_id` is any hashable value (an int, a string, a tuple). `n_tokens` is an int
$\ge 1$.

![The pool after the first four calls of the example. The blocks of sequence `a` are not adjacent, and only its last block has empty slots.](assets/block-table.svg)

## Example

```python
alloc = BlockAllocator(n_blocks=6, block_size=4)

alloc.append("a", 5)     # [0, 1]     a: 5 tokens, blocks 0 and 1 (block 1 holds 1 of 4)
alloc.append("b", 3)     # [2]        b: 3 tokens
alloc.append("a", 3)     # []         a: 8 tokens; the 3 tokens fit in block 1
alloc.append("a", 1)     # [3]        a: 9 tokens; blocks 0 and 1 were full
alloc.block_table("a")   # [0, 1, 3]
alloc.free_blocks()      # 2          (blocks 4 and 5; this is Figure 2)

alloc.free("b")          #            block 2 is free again
alloc.append("c", 9)     # [2, 4, 5]  lowest free ids first; the pool is now empty
alloc.append("a", 4)     # raises OutOfBlocks: a has 3 free slots, needs 1 more block
alloc.block_table("a")   # [0, 1, 3]  unchanged, and a still has 9 tokens

alloc.free("c")
alloc.append("a", 4)     # [2]        a: 13 tokens
alloc.block_table("a")   # [0, 1, 3, 2]
alloc.free_blocks()      # 2
```

The third call shows the rule that gives paging its value: `a` had 5 tokens in two
blocks, so three slots of block 1 were empty, and three more tokens cost no memory.
The refused call shows the other rule. Three of the four tokens would have fit in
block 3, but an append is all or nothing: after the exception `a` still has 9
tokens, and the same call succeeds once a block is free.

## Constraints

- $1 \le$ `n_blocks` $\le 1000$, $1 \le$ `block_size` $\le 64$, and at most a few
  hundred calls per allocator.
- Results are compared exactly: the ids returned by `append`, every block table, and
  `free_blocks()` after every operation.
- Property checks after every operation of random workloads: no block is in two
  block tables (or twice in one), every id is in `[0, n_blocks)`, each live sequence
  owns exactly $\lceil t / \texttt{block\_size} \rceil$ blocks, and `free_blocks()`
  plus the lengths of all block tables equals `n_blocks`.
- `OutOfBlocks` must be an exception class defined by your solution, as in the
  starter.

## Why this matters

Before paging, a server reserved one contiguous slab of KV cache per request, sized
for the longest output the request might produce. Most requests stop far earlier, so
most of that memory held nothing, and the number of requests that fit in a batch,
which is what throughput comes from, was a fraction of what the GPU could hold. vLLM's
PagedAttention (Kwon et al., 2023) borrowed the answer from operating systems: cut
the cache into fixed-size blocks, give each sequence a block table that maps its
logical blocks to physical ones anywhere in the pool, and allocate a block only when
the previous ones are full. The waste per sequence drops to at most one partly
filled block. The allocator is a few lines, and its bugs are the bugs of every
allocator: a block handed to two sequences makes one request attend to another
request's keys and values, a leaked block shrinks the server's capacity until
restart, and an allocation that fails half-way leaves a sequence holding memory that
nothing accounts for.
