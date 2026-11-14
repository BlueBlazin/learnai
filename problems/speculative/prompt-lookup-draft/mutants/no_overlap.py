# BUG: only accepts occurrences that end before the suffix begins (start + n <= length - n), missing occurrences that overlap the suffix, as in runs of a repeated token.
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - 2 * n + 1):
            if context[start : start + n] == suffix:
                return context[start + n : start + n + k]
    return []
