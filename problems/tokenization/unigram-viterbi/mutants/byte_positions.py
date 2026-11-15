# BUG: the dynamic program runs over UTF-8 bytes, so an unknown character of several bytes becomes several unknown pieces (each penalized, none of them a character).
UNK_PENALTY = 10.0


def unigram_encode(text: str, pieces: dict[str, float]) -> list[str]:
    data = text.encode("utf-8")
    table = {piece.encode("utf-8"): score for piece, score in pieces.items()}
    n = len(data)
    unk_score = min(pieces.values()) - UNK_PENALTY
    max_len = max(len(piece) for piece in table)
    best: list[tuple[float, int, int]] = [(0.0, 0, 0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        choice = None
        for length in range(min(max_len, n - i), 0, -1):
            score = table.get(data[i : i + length])
            if score is None:
                if length > 1:
                    continue
                score = unk_score  # one unknown byte
            rest_score, rest_count, _ = best[i + length]
            total, count = score + rest_score, 1 + rest_count
            if choice is None or total > choice[0] or (total == choice[0] and count < choice[1]):
                choice = (total, count, length)
        best[i] = choice
    out = []
    i = 0
    while i < n:
        length = best[i][2]
        out.append(data[i : i + length].decode("utf-8", errors="replace"))
        i += length
    return out
