Implement the block allocator for a paged KV cache. The allocator manages a pool of fixed-size physical blocks and maintains a *block table* for each live sequence listing the blocks that hold its keys and values. A sequence receives a new block only when its existing blocks are full, so no sequence wastes more than one partly filled block.

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

The pool holds `n_blocks` physical blocks with ids `0` through `n_blocks - 1`. Each block has room for the keys and values of `block_size` tokens. Initially every block is free and no sequence exists. The allocator stores no tensors. It only tracks which blocks belong to each sequence.

A **live sequence** has a token count $t \ge 1$ and a block table `table` containing the ids of the blocks it owns, in the order it received them. Token $i$ of the sequence (counting from `0`) lives in slot `i % block_size` of block `table[i // block_size]`. Because slots fill in order, a live sequence always owns exactly $\lceil t / \texttt{block\_size} \rceil$ blocks, and only the last block can have empty slots.

- `append(seq_id, n_tokens)` adds `n_tokens` tokens to sequence `seq_id`. If `seq_id` is not live, this call starts a new sequence with `0` existing tokens. The new tokens first fill any empty slots in the sequence's last block, and only the remaining tokens require new blocks. If the sequence has $t$ tokens before the call and $n = \texttt{n\_tokens}$, the number of new blocks needed is

  $$
  \left\lceil \frac{t + n}{\texttt{block\_size}} \right\rceil - \left\lceil \frac{t}{\texttt{block\_size}} \right\rceil .
  $$

  Take new blocks from the free pool **lowest id first** and append them to the block table in that order. Return the list of newly allocated block ids in allocation order. Return an empty list if all new tokens fit in the last block.

  If the call needs more blocks than are free, raise `OutOfBlocks` and **change nothing**. Do not remove any block from the pool, leave the sequence's token count and block table unchanged, and do not create a live sequence if `seq_id` was not live.
- `free(seq_id)` returns all blocks of the sequence to the pool and removes the sequence. Afterward `seq_id` is no longer live and may be reused later for a new sequence. Return `None`. Raise `KeyError` if `seq_id` is not live.
- `block_table(seq_id)` returns a new list containing the sequence's block table, so mutating the returned list does not affect the allocator. Raise `KeyError` if `seq_id` is not live.
- `free_blocks()` returns the number of free blocks in the pool.

`seq_id` can be any hashable value (such as an `int`, `str`, or `tuple`), and `n_tokens` is an `int` $\ge 1$.

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

In the third call, `a` already holds 5 tokens across two blocks, leaving 3 empty slots in block `1`. Appending 3 more tokens fills those slots and allocates no new blocks.

In the refused call, 3 of the 4 new tokens would fit in block `3`, but the fourth token needs a new block while the pool is empty. Because `append` is all-or-nothing, `a` stays at 9 tokens with table `[0, 1, 3]`, and the same call succeeds once blocks are freed.

## Constraints

- $1 \le \texttt{n\_blocks} \le 1000$, $1 \le \texttt{block\_size} \le 64$, and at most a few hundred calls per allocator.
- Outputs are checked for exact equality, including the ids returned by `append`, every block table, and `free_blocks()` after each operation.
- Property checks run after every operation on random workloads. No block may appear in two block tables or twice in one table. Every block id must lie in `[0, n_blocks)`. Every live sequence must own exactly $\lceil t / \texttt{block\_size} \rceil$ blocks. `free_blocks()` plus the lengths of all block tables must equal `n_blocks`.
- `OutOfBlocks` must be an exception class defined by your solution, as in the starter code.

## Why this matters

Before paged KV caches, serving engines reserved one contiguous slab of memory per request, sized for the maximum output length. Most requests finish much earlier, so most of that reserved memory stayed empty and limited the batch size that could fit on the GPU.

vLLM's PagedAttention (Kwon et al., 2023) adapted virtual memory paging from operating systems. The cache is split into fixed-size physical blocks, each sequence keeps a block table mapping logical blocks to physical ones anywhere in the pool, and a new block is allocated only when previous blocks are full. Waste drops to at most one partly filled block per sequence.

Allocator bugs fail silently in serving engines. Handing the same block to two sequences makes one request attend to another request's keys and values. Leaking a freed block shrinks serving capacity until restart. Failing halfway through an append leaves a sequence holding memory that nothing accounts for.
