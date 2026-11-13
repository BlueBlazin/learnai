from itertools import pairwise


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    rank = {(left, right): i for i, (left, right) in enumerate(merges)}
    ids = list(text.encode("utf-8"))  # byte values are the first 256 token ids
    while len(ids) >= 2:
        # Of all adjacent pairs present right now, the one that appears earliest in the
        # merge list wins. Pairs that are not merges get a rank past the end.
        pair = min(pairwise(ids), key=lambda p: rank.get(p, len(rank)))
        if pair not in rank:
            break  # no adjacent pair is a merge: done
        ids = _replace_pair(ids, pair, 256 + rank[pair])
    return ids


def _replace_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair`, scanning left to right without overlap."""
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2  # both halves are consumed, so "aaa" becomes [aa, a]
        else:
            out.append(ids[i])
            i += 1
    return out


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    # Expand merged tokens back into bytes with an explicit stack (no recursion limit,
    # and no assumption that a merge only refers to earlier merges).
    data = bytearray()
    stack = list(reversed(ids))
    while stack:
        token = stack.pop()
        if token < 256:
            data.append(token)
        else:
            left, right = merges[token - 256]
            stack.append(right)  # pushed first, so `left` is expanded first
            stack.append(left)
    # Decode once, at the end: a character's bytes may be spread over several tokens.
    return data.decode("utf-8", errors="replace")
