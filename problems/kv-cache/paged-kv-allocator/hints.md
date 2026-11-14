How many blocks does a sequence with $t$ tokens own? If you can answer that from $t$
alone, you can also say how many *new* blocks an append of $n$ tokens needs, without
tracking the fill level of the last block separately.

---

Keep three things: the free ids, and per sequence its block table and its token
count. A sequence of $t$ tokens owns $\lceil t / \texttt{block\_size} \rceil$ blocks,
so an append needs $\lceil (t + n) / \texttt{block\_size} \rceil$ minus the length of
the block table. In integer arithmetic, $\lceil a / b \rceil$ is `(a + b - 1) // b`
or `-(-a // b)`. Compare that number with the number of free blocks *before* you
change anything.

---

Order `append` as: compute, check, then commit.

```
total  = tokens.get(seq_id, 0) + n_tokens
needed = ceil(total / block_size) - len(table of seq_id, or [] if new)
if needed > number of free blocks: raise OutOfBlocks      # nothing was touched yet
take the `needed` lowest free ids, extend the table, store total
```

For "lowest free id first", a min-heap (`heapq`) of free ids gives the lowest in
$O(\log n)$; in `free`, push each block back. A sorted list works too. Create the
entry for a new sequence only in the commit step, or a refused append leaves an
empty sequence behind.
