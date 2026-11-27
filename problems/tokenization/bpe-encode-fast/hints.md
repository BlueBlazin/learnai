Each merge changes the sequence at only one spot. Two adjacent tokens become one, so only the pairs touching those two tokens change while the rest of the sequence stays intact. What data structures let you (a) find the lowest-rank pair without scanning every pair, and (b) remove a token from the middle of the sequence without copying the whole list?

---

Store the tokens in a doubly linked list using two integer arrays `prev` and `next` indexed by byte position. Keep the merged token in its left half's slot and unlink the right half's slot. Maintain a min-heap of `(rank, slot)` entries for every adjacent pair that appears in `merges`, where `slot` is the position of the pair's left token. Ordering by rank first and slot second matches the specification's order (lowest rank, then left to right).

Because you cannot delete from the middle of a heap, entries become stale when one of their tokens is merged away. Leave stale entries in the heap instead of trying to remove them. When you pop `(rank, slot)`, check the linked list to confirm that `slot` is still alive and that the token at `slot` and its right neighbor still match `merges[rank]`. If not, discard the entry.

---

Process one rank per pass. Pop every heap entry matching the top rank, apply each valid merge, and record the slots of the newly formed tokens. Only **after** all entries of that rank have been popped, push the new neighbor pairs `(left_neighbor, new_id)` and `(new_id, right_neighbor)` if they exist in `merges`:

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

Pushing inside the inner loop fails when a merge references a token created by a later merge. In that case, a newly created pair can have a lower rank than `r` and would be merged before the remaining occurrences of rank `r` finish.
