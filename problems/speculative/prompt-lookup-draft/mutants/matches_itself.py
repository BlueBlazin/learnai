# BUG: the search runs up to index length - n, so when an n-gram has no earlier occurrence the suffix matches itself; nothing follows it, and the draft is empty instead of falling back to a shorter n-gram.
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    for n in range(min(max_ngram, length), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - n + 1):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []
