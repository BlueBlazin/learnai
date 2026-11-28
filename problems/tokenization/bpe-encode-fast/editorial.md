## The idea

The naive definition performs two full scans per applied merge: one to find the lowest-rank pair and another to replace it. Both scans are unnecessary because merging one pair changes at most three pairs in the sequence. The merged pair itself disappears, and its left and right neighbors form at most two new pairs with the merged token. Every other pair and its rank remain unchanged.

To update only the affected neighborhood, combine two data structures:

- A **doubly linked list** of tokens, so absorbing the right token of a pair takes $O(1)$ time and both neighbors are directly accessible.
- A **min-heap** of `(rank, slot)` entries for every adjacent pair that appears in `merges`, where `slot` is the index of the pair's left token.

The heap ordering matches the definition directly. The smallest entry has the lowest available rank (step 3), and among entries of the same rank, the smallest `slot` is the leftmost occurrence (step 4). Two details ensure exact equivalence with the reference procedure.

**Stale entries.** Because a binary heap cannot delete arbitrary elements in the middle, entries remain in the heap after their pair is invalidated. Whenever `(r, i)` is popped, validate it against the linked list. Slot `i` must still hold the left token of `merges[r]`, and its right neighbor must still hold the right token. This check is sound because tokens only grow through merges. Once slot `i` or its right neighbor stops matching that pair, they can never match it again, so a stale entry can never turn valid later. This also handles runs automatically. In `a a a` with merge `(a, a)`, the heap holds `(0, 0)` and `(0, 1)`. Popping `(0, 0)` merges slots 0 and 1, so when `(0, 1)` is popped next, slot 1 is already marked dead.

**The pass boundary.** The definition applies all occurrences of the chosen rank before selecting the next rank. For tables produced by standard BPE training, this happens automatically because any pair containing token `256 + r` has a rank greater than `r` and sorts after the remaining rank-`r` entries. Here, however, a merge may reference a token created by a later merge. With `merges = [(257, a), (a, b)]` and text `abab`, merging the first `a b` creates `([ab], a)` with rank 0. Pushing it immediately would jump ahead of the second `a b` and consume its `a`.

To preserve the pass boundary, process one rank as a batch. Pop all entries of the top rank, apply every valid merge, and push the newly formed neighbor pairs only after the batch completes. Because no token can contain itself, a merge of rank `r` can never create a new pair of rank `r`, so a batch never needs its own newly created pairs.

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

The linked list is represented with two integer lists (`prev` and `nxt`) indexed by starting byte position. Because a merged token keeps the slot of its left half, a pair's left slot is a stable identifier whose integer order always matches left-to-right text order. Node objects with `prev` and `next` attributes also work and measure about 1.2x slower.

When two newly merged tokens end up adjacent (such as `[ab] [ab]`), the pair between them is pushed twice, once from each side. The second copy is stale by the time it is popped, so no deduplication is required.

## Common mistakes

- **Keeping the linear scans.** The reference solution for `bpe-encode` is correct but quadratic. Finding the next rank with a heap while still rebuilding the entire list per merge is also quadratic. On the speed benchmark, those two approaches measure about 200x and 140x slower than the reference.
- **Applying stale entries.** Checking only whether a slot is still alive is insufficient. On `abc` with `[(b, c), (a, b)]`, slot 0 is still alive after `(b, c)` merges, but its right neighbor is now `[bc]` rather than `b`. Always verify both token ids against `merges[r]`.
- **Pushing new pairs immediately.** Pushing neighbor pairs inside the inner loop works for standard trained tables but fails when a merge references a token from a later merge. Because this bug only surfaces on out-of-order ranks, it is easy to miss if you only test against real tokenizers.
- **Forgetting the left neighbor.** After a merge, both `(left_neighbor, new_id)` and `(new_id, right_neighbor)` can become valid merges. Pushing only the right neighbor handles rightward growth (`ab` then `abc`) but misses merges like `c` + `ab`.
- **Popping occurrences from the right.** Ordering heap entries by `(rank, -slot)` or storing positions on a stack merges `a a a` into `a [aa]` instead of `[aa] a`. The specification scans left to right.

## Complexity

Let $n$ be the number of input bytes and $m$ be the number of merges in the table. Building the rank map, linked list, and initial heap takes $O(n + m)$ time. Each applied merge reduces the token count by 1, so fewer than $n$ merges can occur. Each merge pushes at most two heap entries, and every entry is popped once, so the heap processes at most $3n$ entries in total. Total time is $O(n \log n + m)$ (or $O(n \log n)$ for a fixed table), with $O(n + m)$ memory.

By contrast, the naive scan costs $O(n)$ per applied merge, or $O(n \cdot \min(n, m))$ overall. On the 2,000-byte benchmark, that is already 200x slower than the reference, and the gap grows linearly with input length.

## In the wild

`llama.cpp` implements BPE tokenization this way in `llm_tokenizer_bpe_session::tokenize` (`src/llama-vocab.cpp`). It stores symbols in a vector linked by `prev` and `next` indices, uses a priority queue ordered by rank and left symbol index, skips stale entries whose symbols were absorbed or no longer match, and pushes neighbor bigrams after each merge. It pushes immediately because the trained vocabularies it loads never violate rank order. `tiktoken` uses the linear-scan loop for short pieces (`_byte_pair_merge` in `src/lib.rs`, documented as $O(mn)$), relies on regex pre-tokenization to keep typical pieces short, and switches to a heap-based routine for long pieces.
