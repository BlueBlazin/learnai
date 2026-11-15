# BUG: measures the word's length in UTF-8 bytes instead of characters, so short non-ASCII words are declared too long.
def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]:
    if len(word.encode("utf-8")) > max_chars:
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
            return [vocab[unk]]
    return ids
