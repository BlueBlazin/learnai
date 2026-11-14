def lookup_draft(context: list[int], max_ngram: int, min_ngram: int, k: int) -> list[int]:
    """Draft tokens by looking up the end of `context` in its own past.

    context: the token ids so far (prompt and generated tokens); not modified.
    max_ngram, min_ngram: n-gram lengths to try, longest first, 1 <= min_ngram <= max_ngram.
    k: the maximum number of tokens to propose, >= 1.
    For n from max_ngram down to min_ngram: if the last n tokens occur earlier in the
    context (anywhere except as the suffix itself; overlap is fine), take the
    earliest such occurrence and return up to k tokens that followed it.
    Returns a new list of 0 to k ints; [] when nothing matches.
    """
    raise NotImplementedError
