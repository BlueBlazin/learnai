# BUG: scans start indices from the right and returns the continuation of the most recent occurrence instead of the earliest one.
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - n - 1, -1, -1):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []
