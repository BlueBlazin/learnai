def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    length = len(context)
    # Longest n-gram first. An n-gram of the whole context (n >= length) has no room
    # for an earlier occurrence, so start at length - 1.
    for n in range(min(max_ngram, length - 1), min_ngram - 1, -1):
        suffix = context[length - n :]
        # Start indices from left to right: the first hit is the earliest occurrence.
        # Stopping before length - n excludes the suffix itself and nothing else, so
        # occurrences that overlap the suffix are found.
        for start in range(length - n):
            if context[start : start + n] == suffix:
                # Slicing stops at the end of the context: the draft may be shorter than k.
                return context[start + n : start + n + k]
    return []
