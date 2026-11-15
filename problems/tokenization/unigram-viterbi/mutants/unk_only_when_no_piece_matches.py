# BUG: the unknown piece is offered only at positions where no vocabulary piece starts, instead of wherever the character itself is not in the vocabulary.
UNK_PENALTY = 10.0


def unigram_encode(text: str, pieces: dict[str, float]) -> list[str]:
    n = len(text)  # positions are characters (code points), not bytes
    unk_score = min(pieces.values()) - UNK_PENALTY
    max_len = max(len(piece) for piece in pieces)
    # best[i] describes the best segmentation of the suffix text[i:]:
    # (total score, number of pieces, length of its first piece).
    best: list[tuple[float, int, int]] = [(0.0, 0, 0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        choice = None
        # Longest candidate first, and a later candidate must be strictly better, so
        # among equally good segmentations the one with the longer first piece stays.
        for length in range(min(max_len, n - i), 0, -1):
            score = pieces.get(text[i : i + length])
            if score is None:
                if length > 1 or choice is not None:
                    continue
                score = unk_score  # used only when no piece starts here
            rest_score, rest_count, _ = best[i + length]
            total, count = score + rest_score, 1 + rest_count
            if choice is None or total > choice[0] or (total == choice[0] and count < choice[1]):
                choice = (total, count, length)
        best[i] = choice
    out = []
    i = 0
    while i < n:
        length = best[i][2]
        out.append(text[i : i + length])
        i += length
    return out
