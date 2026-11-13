# BUG: tokenizes by greedy longest match against the vocabulary (WordPiece style) instead of applying merges in rank order; it can emit a long token that the merge order never builds.
def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    data = text.encode("utf-8")
    vocab = {bytes([b]): b for b in range(256)}
    for i in range(len(merges)):
        vocab.setdefault(_token_bytes(256 + i, merges), 256 + i)
    longest = max(len(token) for token in vocab)
    ids = []
    pos = 0
    while pos < len(data):
        for length in range(min(longest, len(data) - pos), 0, -1):
            token = data[pos : pos + length]
            if token in vocab:
                ids.append(vocab[token])
                pos += length
                break
    return ids


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
    data = b"".join(_token_bytes(token, merges) for token in ids)
    return data.decode("utf-8", errors="replace")
