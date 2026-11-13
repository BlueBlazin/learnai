# BUG: decode converts each token's bytes to text separately and joins the pieces; a character whose UTF-8 bytes are split across tokens turns into U+FFFD replacement characters.
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
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _token_bytes(token: int, merges: list[tuple[int, int]]) -> bytes:
    data = bytearray()
    stack = [token]
    while stack:
        token = stack.pop()
        if token < 256:
            data.append(token)
        else:
            left, right = merges[token - 256]
            stack.append(right)
            stack.append(left)
    return bytes(data)


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    return "".join(_token_bytes(t, merges).decode("utf-8", errors="replace") for t in ids)
