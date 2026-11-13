# BUG: applies each merge once, in list order, in a single sweep over the table; a merge whose pair only appears after a later merge has run is never applied.
def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    ids = list(text.encode("utf-8"))
    for i, (left, right) in enumerate(merges):
        ids = _replace_pair(ids, (left, right), 256 + i)
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
