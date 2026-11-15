# BUG: takes the longest vocabulary piece at each position (WordPiece-style greedy matching) instead of the segmentation with the highest total score.
UNK_PENALTY = 10.0


def unigram_encode(text: str, pieces: dict[str, float]) -> list[str]:
    n = len(text)
    max_len = max(len(piece) for piece in pieces)
    out = []
    i = 0
    while i < n:
        length = 1  # a character that starts no piece stands for itself
        for candidate in range(min(max_len, n - i), 0, -1):
            if text[i : i + candidate] in pieces:
                length = candidate
                break
        out.append(text[i : i + length])
        i += length
    return out
