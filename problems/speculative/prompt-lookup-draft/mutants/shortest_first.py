# BUG: tries n-gram lengths from min_ngram upward, so a short, coincidental match is preferred over a longer one.
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    for n in range(min_ngram, min(max_ngram, length - 1) + 1):
        suffix = context[length - n :]
        for start in range(length - n):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []
