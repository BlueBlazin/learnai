# BUG: skips an occurrence that is followed by fewer than k tokens and moves on to a later one (or to a shorter n-gram), instead of returning the shorter draft of the earliest occurrence.
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - n):
            if context[start : start + n] == suffix and start + n + k <= length:
                return context[start + n : start + n + k]
    return []
