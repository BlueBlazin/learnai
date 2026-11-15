def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]:
    if len(word) > max_chars:
        return [vocab[unk]]  # too long to split: one unknown token
    ids = []
    start = 0
    while start < len(word):
        # Longest candidate first. Pieces after the first carry the "##" prefix.
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                ids.append(vocab[piece])
                start = end
                break
        else:
            # Nothing matches here: the whole word is unknown, and the pieces found
            # so far are dropped. No shorter earlier piece is tried.
            return [vocab[unk]]
    return ids
