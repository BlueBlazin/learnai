An append only ever writes into one existing block of a sequence: its last one, and
only if that block has room. So "may I write here?" is a question about a single
block. What do you need to know about that block, and when is the answer computed:
at the fork, or at the append?

---

Keep a reference count per block next to the free heap. `fork` copies the parent's
table (a new list with the same ids) and increments the count of every block in it.
`free` decrements, and pushes a block back to the pool only when its count reaches
0. In `append`, the copy is needed exactly when `tokens % block_size != 0` and
`refs[table[-1]] > 1`, evaluated now. Add that 0 or 1 to the number of growth
blocks *before* comparing with the number of free blocks, so that a refusal has
nothing to undo.

---

```
copy   = 1 if tokens % block_size != 0 and refs[table[-1]] > 1 else 0
growth = ceil((tokens + n) / block_size) - len(table)
if copy + growth > number of free blocks: raise OutOfBlocks     # nothing touched
new = the (copy + growth) lowest free ids; refs[b] = 1 for each
if copy:
    src = table[-1]
    table = table[:-1] + [new[0]]      # in a new list: the sibling keeps src
    refs[src] -= 1                     # src still has other holders; it is not freed
    copies = [(src, new[0])]
table = table + new[copy:]
return new, copies
```

Build the new table as a new list. If parent and child ever point at the same list
object, or you edit a table in place while a fork still refers to it, one
sequence's blocks appear in the other's table.
