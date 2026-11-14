# BUG: the copy starts at the last token of the matched n-gram (start + n - 1) instead of the token after it, so the draft begins with a token the context already ends with.
def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        for start in range(length - n):
            if context[start : start + n] == suffix:
                return context[start + n - 1 : start + n - 1 + k]
    return []
