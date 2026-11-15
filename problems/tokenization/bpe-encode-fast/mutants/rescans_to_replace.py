# BUG: correct but still quadratic: a heap picks the next rank in O(log n), but each merge is then applied by rebuilding the whole sequence, and the new pairs are found by scanning it again.
import heapq
from itertools import pairwise


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    rank = {(left, right): i for i, (left, right) in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    heap = list({rank[pair] for pair in pairwise(ids) if pair in rank})
    heapq.heapify(heap)
    while heap:
        r = heapq.heappop(heap)
        new_id = 256 + r
        replaced = _replace_pair(ids, merges[r], new_id)
        if len(replaced) == len(ids):
            continue  # the pair is no longer in the sequence
        ids = replaced
        for found in {rank[pair] for pair in pairwise(ids) if new_id in pair and pair in rank}:
            heapq.heappush(heap, found)
    return ids


def _replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
