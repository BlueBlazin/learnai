# BUG: after a merge only the pair (new token, right neighbor) is pushed; the pair (left neighbor, new token) never enters the heap, so merges that extend a token to the left are missed.
import heapq
from itertools import pairwise


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    rank = {(left, right): i for i, (left, right) in enumerate(merges)}
    ids = list(text.encode("utf-8"))  # ids[i] is the token that starts at slot i, or -1
    n = len(ids)
    # A doubly linked list over the slots: a merge removes one slot in O(1).
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1))  # n means "no next slot"
    # One heap entry (rank, slot of the left token) per mergeable adjacent pair.
    heap = [(rank[pair], i) for i, pair in enumerate(pairwise(ids)) if pair in rank]
    heapq.heapify(heap)
    while heap:
        r = heap[0][0]
        left, right = merges[r]
        new_id = 256 + r
        merged = []
        # One pass of the definition: every occurrence of the lowest-rank pair, from
        # left to right. Entries of equal rank leave the heap in slot order.
        while heap and heap[0][0] == r:
            i = heapq.heappop(heap)[1]
            j = nxt[i]
            if ids[i] != left or j == n or ids[j] != right:
                continue  # stale: an earlier merge consumed or changed one of the two
            ids[i] = new_id
            ids[j] = -1  # the right token is absorbed; unlink its slot
            nxt[i] = nxt[j]
            if nxt[j] < n:
                prev[nxt[j]] = i
            merged.append(i)
        # Only now do the pairs around the new tokens enter the heap. Pushing them
        # inside the loop would let a new pair of lower rank interrupt the pass.
        for i in merged:
            j = nxt[i]
            if j < n and (new_id, ids[j]) in rank:
                heapq.heappush(heap, (rank[new_id, ids[j]], i))
    return [token for token in ids if token >= 0]
