# BUG: the dynamic program runs forward over prefixes and keeps the longest last piece among equal candidates, so ties go to the segmentation whose later pieces are longer ("a", "bc" instead of "ab", "c").
UNK_PENALTY = 10.0


def unigram_encode(text: str, pieces: dict[str, float]) -> list[str]:
    n = len(text)
    unk_score = min(pieces.values()) - UNK_PENALTY
    max_len = max(len(piece) for piece in pieces)
    # best[j] describes the best segmentation of the prefix text[:j]:
    # (total score, number of pieces, length of its last piece).
    best: list[tuple[float, int, int]] = [(0.0, 0, 0)] * (n + 1)
    for j in range(1, n + 1):
        choice = None
        for length in range(min(max_len, j), 0, -1):
            score = pieces.get(text[j - length : j])
            if score is None:
                if length > 1:
                    continue
                score = unk_score
            rest_score, rest_count, _ = best[j - length]
            total, count = score + rest_score, 1 + rest_count
            if choice is None or total > choice[0] or (total == choice[0] and count < choice[1]):
                choice = (total, count, length)
        best[j] = choice
    out = []
    j = n
    while j > 0:
        length = best[j][2]
        out.append(text[j - length : j])
        j -= length
    return out[::-1]
