An append only ever writes into one existing block of a sequence: its last block, and only when that block still has empty slots. What do you need to know about that single block before writing to it, and should you check that condition at `fork` time or at `append` time?

---

Store a reference count for each block alongside the free heap. In `fork`, copy the parent's block table into a new list and increment the reference count of every block in it. In `free`, decrement each block's count and return a block to the pool only when its count reaches `0`. In `append`, a copy is needed if and only if `tokens % block_size != 0` and `refs[table[-1]] > 1` right now. Add that `0` or `1` to the number of growth blocks *before* checking against `free_blocks()`, so a refused append leaves all state unchanged.

---

```
copy   = 1 if tokens % block_size != 0 and refs[table[-1]] > 1 else 0
growth = ceil((tokens + n_tokens) / block_size) - len(table)
if copy + growth > number of free blocks: raise OutOfBlocks     # nothing touched
new = the (copy + growth) lowest free ids; refs[b] = 1 for each
if copy:
    src = table[-1]
    table = table[:-1] + [new[0]]      # new list so siblings keep src
    refs[src] -= 1                     # src still has other holders, so do not free it
    copies = [(src, new[0])]
table = table + new[copy:]
return new, copies
```

Always give each forked sequence its own list for its block table. If parent and child share the same list object, mutating one table in place corrupts the other.
