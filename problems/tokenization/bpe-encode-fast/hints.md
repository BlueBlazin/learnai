A merge changes the sequence in one place: two tokens become one, and the only pairs
that change are the ones that touched those two tokens. Everything else in the
sequence is as it was. What do you need to store so that (a) the lowest-rank pair
present is found without looking at every pair, and (b) removing a token from the
middle of the sequence does not cost a copy of the sequence?

---

Keep the tokens in a doubly linked list (two integer arrays `prev` and `next` over the
byte positions are enough; a merged token lives in the slot of its left half and the
right half's slot is unlinked). Keep a min-heap of `(rank, slot)` entries, one per
adjacent pair that is a merge, where `slot` is the position of the pair's left token.
The heap's order, rank first and slot second, is exactly the order of the definition:
lowest rank, and within a rank from left to right.

You cannot delete from the middle of a heap, so entries go stale: the pair they were
pushed for may be gone by the time they are popped. Do not try to remove them. When an
entry comes out, look at the list: is the slot still alive, and are the token there
and its right neighbor still the two tokens of that rank's pair? If not, drop the
entry.

---

One pass of the definition is: pop every entry whose rank equals the rank on top of
the heap, in order, and merge at each one that is still valid. Collect the slots of
the new tokens, and push the pairs around them, `(left neighbor, new)` and `(new,
right neighbor)`, only **after** the last entry of that rank has been popped:

```python
while heap:
    r = heap[0][0]
    merged = []
    while heap and heap[0][0] == r:
        i = heapq.heappop(heap)[1]
        ...  # skip if stale; otherwise merge slot i with its right neighbor
        merged.append(i)
    for i in merged:
        ...  # push the pairs around slot i that are in the table
```

Pushing inside the inner loop is the natural thing to write and is wrong when a merge
refers to a token of a later merge: the new pair can have a lower rank than `r`, and
it would be merged before the remaining occurrences of the current pair.
