# BUG: when no piece matches, keeps the pieces found so far and appends one unknown token for the rest, instead of replacing the whole word.
def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]:
    if len(word) > max_chars:
        return [vocab[unk]]
    ids = []
    start = 0
    while start < len(word):
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                ids.append(vocab[piece])
                start = end
                break
        else:
            ids.append(vocab[unk])
            break
    return ids
