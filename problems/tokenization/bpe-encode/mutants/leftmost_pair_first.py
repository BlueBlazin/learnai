# BUG: merges the first mergeable pair found scanning the text from the left, instead of the pair with the lowest rank anywhere in the sequence.
from itertools import pairwise


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    rank = {(left, right): i for i, (left, right) in enumerate(merges)}
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pair = next((p for p in pairwise(ids) if p in rank), None)
        if pair is None:
            break
        ids = _replace_pair(ids, pair, 256 + rank[pair])
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
