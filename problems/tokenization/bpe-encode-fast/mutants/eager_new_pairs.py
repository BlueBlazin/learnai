# BUG: the pairs around a new token are pushed as soon as it is created, so a new pair of lower rank is merged before the remaining occurrences of the current pair (wrong when a merge refers to a token of a later merge).
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
        r, i = heapq.heappop(heap)
        left, right = merges[r]
        new_id = 256 + r
        j = nxt[i]
        if ids[i] != left or j == n or ids[j] != right:
            continue  # stale: an earlier merge consumed or changed one of the two
        ids[i] = new_id
        ids[j] = -1  # the right token is absorbed; unlink its slot
        nxt[i] = nxt[j]
        if nxt[j] < n:
            prev[nxt[j]] = i
        # The pairs around the new token go straight into the heap.
        h = prev[i]
        if h >= 0 and (ids[h], new_id) in rank:
            heapq.heappush(heap, (rank[ids[h], new_id], h))
        j = nxt[i]
        if j < n and (new_id, ids[j]) in rank:
            heapq.heappush(heap, (rank[new_id, ids[j]], i))
    return [token for token in ids if token >= 0]
