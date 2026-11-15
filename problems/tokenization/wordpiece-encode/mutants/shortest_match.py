# BUG: tries the shortest candidate first, so it takes the first single character that is in the vocabulary instead of the longest piece.
def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]:
    if len(word) > max_chars:
        return [vocab[unk]]
    ids = []
    start = 0
    while start < len(word):
        for end in range(start + 1, len(word) + 1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                ids.append(vocab[piece])
                start = end
                break
        else:
            return [vocab[unk]]
    return ids
