## The idea

The definition does two scans per applied merge: one to find the lowest-rank pair and
one to replace it. Both are wasteful for the same reason. A merge at one place changes
at most three pairs: the pair itself disappears, and the pairs with the left and the
right neighbor are replaced by pairs with the new token. Every other pair in the
sequence, and its rank, is untouched. So keep the set of mergeable pairs in a
structure that is updated locally:

- a **doubly linked list** of the tokens, so that absorbing the right half of a pair
  is $O(1)$ and the neighbors of any token are known;
- a **min-heap** of `(rank, slot)` entries, one for every adjacent pair that is in the
  table, where `slot` identifies the pair's left token.

The heap order is the definition's order. The smallest entry has the lowest rank
present (step 3), and among entries of that rank the smallest slot is the leftmost
occurrence (step 4). Two details make the heap agree with the definition exactly.

**Stale entries.** A heap cannot delete from the middle, so an entry stays in it after
its pair is destroyed. When an entry `(r, i)` is popped, check it against the list:
slot `i` must still hold the left token of `merges[r]` and its right neighbor the
right token. This check is sound because tokens only ever grow: once slot `i` has
stopped being the left token of that pair, or its neighbor has stopped being the
right token, they never go back, so a stale entry can never look valid again. It also
handles runs for free. In `a a a` with the merge `(a, a)` the entries are `(0, 0)` and
`(0, 1)`; the first merges slots 0 and 1, and the second finds slot 1 dead.

**The pass boundary.** The definition applies the chosen merge everywhere and only
then chooses again. If the table is the output of BPE training this needs no care: a
pair that contains token `256 + r` can only have a rank above `r`, so a pair created
while rank `r` is being applied always sorts after the remaining rank-`r` entries. The
statement does not promise that. With `merges = [(257, a), (a, b)]` and the text
`abab`, merging the first `a b` creates `([ab], a)`, which has rank 0 and would jump
the queue, consuming the `a` of the second `a b`. The fix is to treat one rank as one
batch: pop all entries of the top rank, merge at the valid ones, and only afterwards
push the pairs around the new tokens. Those pushes look at the list as it is after the
pass, which is what step 2 of the definition does.

No pair created during a pass can have the rank being applied (it contains the token
`256 + r`, and a pair containing its own result would make the token contain itself),
so a batch never needs its own output.

## Reference solution (Python)

```python
import heapq
from itertools import pairwise


def encode(text, merges):
    rank = {(left, right): i for i, (left, right) in enumerate(merges)}
    ids = list(text.encode("utf-8"))  # ids[i] is the token that starts at slot i, or -1
    n = len(ids)
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))  # n means "no next slot"
    heap = [(rank[pair], i) for i, pair in enumerate(pairwise(ids)) if pair in rank]
    heapq.heapify(heap)
    while heap:
        r = heap[0][0]
        left, right = merges[r]
        new_id = 256 + r
        merged = []
        while heap and heap[0][0] == r:
            i = heapq.heappop(heap)[1]
            j = nxt[i]
            if ids[i] != left or j == n or ids[j] != right:
                continue  # stale
            ids[i] = new_id
            ids[j] = -1
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i
            merged.append(i)
        for i in merged:
            h = prev[i]
            if h >= 0 and (ids[h], new_id) in rank:
                heapq.heappush(heap, (rank[ids[h], new_id], h))
            j = nxt[i]
            if j < n and (new_id, ids[j]) in rank:
                heapq.heappush(heap, (rank[new_id, ids[j]], i))
    return [token for token in ids if token >= 0]
```

The linked list is two integer lists indexed by byte position. A merged token keeps
the slot of its left half, so the slot of a pair's left token is a stable name for the
pair, and slot order is text order for as long as the slots live. Node objects with
`prev` and `next` attributes work equally well and measure about 1.2x slower.

When two new tokens end up adjacent (`[ab] [ab]`), the pair between them is pushed
twice, once from each side. The second copy is stale by the time it is popped, so
nothing needs to deduplicate it.

## Common mistakes

- **Keeping the scans.** The reference solution of `bpe-encode` is correct and
  quadratic. So is the half-way version that finds the next rank with a heap and then
  rebuilds the whole sequence to apply it: the selection was never the expensive
  half alone. The speed case measures 200x and 140x for the two.
- **Applying stale entries.** Checking only that the slot is alive is not enough. In
  `abc` with `[(b, c), (a, b)]`, the entry for `(a, b)` at slot 0 is popped after
  `(b, c)` has merged; slot 0 is alive and has a right neighbor, but that neighbor is
  now `[bc]`. Compare both tokens with the pair of the entry's rank.
- **Pushing new pairs immediately.** Correct for trained tables, wrong when a merge
  refers to a later one (see above). The symptom is one token too many or too few on
  inputs with shuffled ranks only, which is why it survives testing against a real
  tokenizer.
- **Forgetting the left neighbor.** After a merge, both `(left neighbor, new)` and
  `(new, right neighbor)` may be merges. Pushing only the right one handles tokens
  that grow rightward, `ab` then `abc`, and misses `c` + `ab`.
- **Popping occurrences from the right.** Entries ordered by `(rank, -slot)`, or
  positions kept on a stack, merge a run `a a a` into `a [aa]`. The definition scans
  from the left.

## Complexity

Let $n$ be the number of bytes. Building the list and the heap is $O(n)$. Every merge
removes one token, so there are fewer than $n$ merges; each pushes at most two entries
and each entry is popped once, so the heap sees at most $3n$ entries and the total
time is $O(n \log n)$, with $O(n + m)$ memory for a table of $m$ merges. The
definition costs $O(n)$ per applied merge, $O(n \cdot \min(n, m))$ in total: on the
speed case's 2,000 bytes that is already 200 times the reference, and the factor grows
linearly with the length of the text.

## In the wild

`llama.cpp` tokenizes BPE vocabularies exactly this way (`llm_tokenizer_bpe_session::
tokenize` in `src/llama-vocab.cpp`): the symbols of a word are a vector linked by
`prev` and `next` indices, the work queue is a priority queue of bigrams ordered by
rank and then by the index of the left symbol, a popped bigram is skipped if either
symbol has been absorbed or if the two symbols no longer spell the bigram's text, and
after a merge the bigrams with the left and the right neighbor are pushed. It pushes
them immediately, which is right for the trained tables it loads, where the pass
boundary cannot matter. `tiktoken` keeps the scan per merge for ordinary pieces
(`_byte_pair_merge` in `src/lib.rs`, whose comment gives the cost as $O(mn)$) and
relies on its regular-expression splitter to keep pieces short, with a heap-based
routine for long ones.
