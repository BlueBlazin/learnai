Extend the paged KV allocator so that sequences can share physical blocks. `fork` creates a new sequence whose block table points to its parent's blocks without allocating memory, and `append` copies a block only when a sequence is about to write into a block that another sequence also holds. Track a reference count for each block so you always know whether a block is still shared at the moment of an append or free.

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

The pool, block tables, and token placement match `paged-kv-allocator`. The pool has `n_blocks` physical blocks with ids `0` through `n_blocks - 1`, each holding up to `block_size` tokens. A live sequence with $t \ge 1$ tokens has a block table `table` of length $\lceil t / \texttt{block\_size} \rceil$, where token $i$ (counting from `0`) lives in slot `i % block_size` of block `table[i // block_size]`. Unlike the basic allocator, one block id may now appear in the block tables of several sequences.

The **reference count** of a block is the number of live sequences whose block table contains it. A block is free if and only if its reference count is `0`.

- `fork(parent_id, child_id)` creates `child_id` with the same token count and a copy of `parent_id`'s block table (the same block ids, stored in its own list). Every block in the table gains one reference. No physical block is allocated, so `fork` never raises `OutOfBlocks`. Raise `ValueError` if `child_id` is already live (including when `child_id == parent_id`). Otherwise raise `KeyError` if `parent_id` is not live. On either error, change nothing. After a fork, parent and child are equal. Either may append, fork again, or be freed first.

- `append(seq_id, n_tokens)` adds $n = \texttt{n\_tokens} \ge 1$ tokens to `seq_id`. If `seq_id` is not live, this call starts a new sequence with `0` existing tokens. Let $t$ be the sequence's token count before the call.

  1. **Copy on write.** The call writes into the sequence's last block if that block has empty slots ($t \bmod \texttt{block\_size} \ne 0$). If that block's reference count is greater than `1`, the sequence must not write into it. Allocate a new block `dst`, replace the last entry `src` of **this sequence's** table with `dst`, and decrement `src`'s reference count. The serving engine will copy the contents of `src` to `dst` before writing the new tokens. A full last block is never written again, so it is never copied regardless of its reference count.
  2. **Growth.** Allocate
     $\lceil (t + n) / \texttt{block\_size} \rceil - \lceil t / \texttt{block\_size} \rceil$
     additional blocks and append them to the table.

  Take all needed blocks from the free pool **lowest id first**, allocating the copy destination (if any) before the growth blocks. Return a tuple `(allocated, copies)`:

  - `allocated` is the list of all block ids allocated by this call, in allocation order (`dst` is `allocated[0]` when a copy occurs).
  - `copies` is `[(src, dst)]` if the call triggered a copy, or `[]` otherwise. Each pair is a tuple of two ints.

  If the call needs more blocks than are free (counting both the copy and growth blocks), raise `OutOfBlocks` and **change nothing**.

- `free(seq_id)` removes the sequence. Decrement the reference count of every block in its table, and return any block whose count reaches `0` to the free pool. Return `None`. Raise `KeyError` if `seq_id` is not live.
- `block_table(seq_id)` returns the sequence's block table as a new list. Raise `KeyError` if `seq_id` is not live.
- `ref_count(block)` returns the reference count of `block` ($0 \le \texttt{block} < \texttt{n\_blocks}$), which is `0` for a free block.
- `free_blocks()` returns the number of blocks with reference count `0`.

`seq_id` can be any hashable value.

> [!NOTE]
> Whether an append triggers a copy depends on the reference count **at the time of the append**. After a fork of two sequences, the first sequence to append copies the shared last block. The other sequence is then its sole holder and writes in place.

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

The third call shows copy-on-write in action. Sequences `a` and `b` share tokens `0` through `5`, and diverge at token `6`. Because `a` still holds block `1`, `b` cannot write its token `6` into slot `2` of block `1`. Instead, `b` allocates block `2`, the engine copies block `1`'s two valid slots into block `2`, and `b` writes into slot `2` of block `2`. Block `0` is full and stays shared for as long as either sequence is live.

`free("b")` shows that reference counts, not original ownership, decide when blocks return to the pool. Although `b` allocated blocks `2` and `3`, `c` still references them, so neither block is freed.

![The pool at four points of the example, with each block's reference count and holders.](assets/pool-steps.svg)

## Constraints

- $1 \le \texttt{n\_blocks} \le 1000$, $1 \le \texttt{block\_size} \le 64$, and at most a few hundred calls per allocator.
- Outputs are checked for exact equality, including the tuple returned by `append`, every block table, every reference count, and `free_blocks()` after each operation.
- Property checks run after every operation on scripted and random workloads:
  - For every block id `b`, `ref_count(b)` must equal the number of live block tables containing `b`, and `free_blocks()` plus the number of distinct blocks in live tables must equal `n_blocks`.
  - *Isolation.* The tests simulate the serving engine by storing token contents in each physical block, applying every copy returned by `append`, writing new tokens to the slots named by each sequence's block table, and reading every live sequence back. Each sequence must read back its ancestors' tokens up to each fork, followed by its own tokens.
- **Copies and allocations are counted.** Four sequences sharing a 10-token prompt (`block_size = 4`, so two full blocks and one half-filled block) that each decode 7 tokens must trigger at most **3** copies and **14** block allocations. That accounts for 3 blocks for the prompt, 1 copy for each of the first three sequences to decode (leaving the fourth as the sole holder of the prompt's last block), and 2 growth blocks per sequence. Copying eagerly on fork would take 20 allocations.
- `OutOfBlocks` must be an exception class defined by your solution, as in the starter code.

## Why this matters

A single prompt often feeds multiple sequences, such as `n > 1` completions for one request, beams in beam search, or branches in tree search and speculative decoding. The shared prompt usually accounts for most of the KV cache memory. With reference-counted blocks, the prompt is stored once, forking costs zero memory and zero copying, and at most one partly filled block is ever copied when sequences diverge.

This is how PagedAttention makes parallel sampling and beam search affordable, and the same mechanism lets a prefix cache share a cached system prompt across requests. Bugs in shared ownership fail silently. A missed copy lets one request overwrite another's attention state. Freeing a block while a sibling still reads it hands live memory to a new request. A reference count that is never decremented leaks capacity until restart.
