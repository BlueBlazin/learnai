# BUG: adds the ## prefix to every candidate, also at the start of the word, so word-initial pieces are looked up as continuation pieces.
def wordpiece(word: str, vocab: dict[str, int], unk: str, max_chars: int) -> list[int]:
    if len(word) > max_chars:
        return [vocab[unk]]
    ids = []
    start = 0
    while start < len(word):
        for end in range(len(word), start, -1):
            piece = "##" + word[start:end]
            if piece in vocab:
                ids.append(vocab[piece])
                start = end
                break
        else:
            return [vocab[unk]]
    return ids
