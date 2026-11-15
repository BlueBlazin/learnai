class SuffixIndex:
    """An online index over a growing token sequence, for prompt-lookup drafting.

    The sequence is every token passed to `extend`, in order: the prompt first, then
    each generated token. After any number of calls, the *match* is the longest suffix
    of the sequence that also occurs earlier in it (overlap allowed).
    """

    def __init__(self) -> None:
        raise NotImplementedError

    def extend(self, token: int) -> None:
        """Append one token to the sequence. Amortized O(1)."""
        raise NotImplementedError

    def match_length(self) -> int:
        """The length of the longest suffix that has an earlier occurrence (0 if none)."""
        raise NotImplementedError

    def draft(self, max_len: int, min_match: int = 1) -> list[int]:
        """The tokens that followed the earliest occurrence of the match.

        max_len: the maximum number of draft tokens, at least 1.
        min_match: the shortest match worth drafting from, at least 1.
        Returns a new list of 0 to max_len Python ints: empty when the match is shorter
        than min_match, otherwise the tokens after the earliest earlier occurrence of
        the match, up to the end of the sequence. O(max_len).
        """
        raise NotImplementedError
