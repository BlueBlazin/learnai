How many blocks does a sequence with $t$ tokens own? If you can compute that directly from $t$, you can also compute how many *new* blocks an append of `n_tokens` needs without tracking empty slots in the last block separately.

---

Keep three pieces of state: the free block ids, each live sequence's block table, and each live sequence's token count. A sequence with $t$ tokens owns $\lceil t / \texttt{block\_size} \rceil$ blocks, so appending $n = \texttt{n\_tokens}$ tokens needs $\lceil (t + n) / \texttt{block\_size} \rceil$ minus the current length of its block table. In integer arithmetic, $\lceil a / b \rceil$ is `(a + b - 1) // b` or `-(-a // b)`. Check that enough free blocks exist *before* changing any state.

---

Structure `append` as compute, check, then commit:

```
total  = tokens.get(seq_id, 0) + n_tokens
needed = ceil(total / block_size) - len(table of seq_id, or [] if new)
if needed > number of free blocks: raise OutOfBlocks      # nothing was touched yet
take the `needed` lowest free ids, extend the table, store total
```

A min-heap (`heapq`) of free ids returns the lowest id in $O(\log \texttt{n\_blocks})$ time, and `free` pushes each released block id back into the heap. A sorted list works as well. Only insert a new `seq_id` into your dictionaries during the commit step so a refused append never leaves an empty sequence behind.
