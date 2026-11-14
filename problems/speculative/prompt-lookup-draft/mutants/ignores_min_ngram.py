# BUG: keeps shortening the n-gram down to length 1 instead of stopping at min_ngram.
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    for n in range(min(max_ngram, length - 1), 0, -1):
        suffix = context[length - n :]
        for start in range(length - n):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []
