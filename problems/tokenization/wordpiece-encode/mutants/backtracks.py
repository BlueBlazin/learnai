# BUG: when a later position fails, goes back and tries a shorter earlier piece; WordPiece is greedy and returns the unknown token instead.
def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]:
    if len(word) > max_chars:
        return [vocab[unk]]

    def split(start: int) -> list[int] | None:
        if start == len(word):
            return []
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                rest = split(end)
                if rest is not None:
                    return [vocab[piece], *rest]
        return None

    ids = split(0)
    return [vocab[unk]] if ids is None else ids
