# BUG: replaces the pair at every position where it starts, including positions that overlap the previous match, so a run like "aaa" with the merge (a, a) becomes two merged tokens.
from itertools import pairwise


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    rank = {(left, right): i for i, (left, right) in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pair = min(pairwise(ids), key=lambda p: rank.get(p, len(rank)))
        if pair not in rank:
            break
        ids = _replace_pair(ids, pair, 256 + rank[pair])
    return ids


def _replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out = []
    for i in range(len(ids)):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)  # a match starts here
        elif i > 0 and (ids[i - 1], ids[i]) == pair:
            continue  # second half of the match that started at i - 1
        else:
            out.append(ids[i])
    return out


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    data = bytearray()
    stack = list(reversed(ids))
    while stack:
        token = stack.pop()
        if token < 256:
            data.append(token)
        else:
            left, right = merges[token - 256]
            stack.append(right)
            stack.append(left)
    return data.decode("utf-8", errors="replace")
