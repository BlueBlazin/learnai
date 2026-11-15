# BUG: never splits a state: the new state links to the transition's target even when that target also holds longer strings that are not suffixes of the sequence, so the match length is overstated and the draft comes from the wrong place.
class SuffixIndex:
    def __init__(self) -> None:
        self._tokens: list[int] = []
        # The suffix automaton of the sequence. State 0 is the empty string. Per state:
        # the transitions, the suffix link, the length of the longest substring in the
        # state, and the earliest position at which one of its substrings ends.
        self._next: list[dict[int, int]] = [{}]
        self._link = [-1]
        self._length = [0]
        self._first_end = [-1]
        self._last = 0  # the state of the whole sequence

    def _state(self, length: int, first_end: int, link: int, nxt: dict[int, int]) -> int:
        self._next.append(nxt)
        self._link.append(link)
        self._length.append(length)
        self._first_end.append(first_end)
        return len(self._length) - 1

    def extend(self, token: int) -> None:
        nxt, link = self._next, self._link
        pos = len(self._tokens)
        self._tokens.append(token)
        # The new state holds the suffixes that occur only here, ending at `pos`.
        cur = self._state(pos + 1, pos, 0, {})
        # Walk the suffixes of the old sequence, longest first: every one that was
        # never followed by `token` now is, and that continuation is new.
        p = self._last
        while p != -1 and token not in nxt[p]:
            nxt[p][token] = cur
            p = link[p]
        if p != -1:
            q = nxt[p][token]
            # (No split: q is linked to as it is.)
            link[cur] = q
        self._last = cur

    def match_length(self) -> int:
        # The suffix link of the whole sequence is the state of its longest suffix
        # that ends in more than one place: the longest one that occurred earlier.
        return self._length[self._link[self._last]] if self._tokens else 0

    def draft(self, max_len: int, min_match: int = 1) -> list[int]:
        if not self._tokens:
            return []
        state = self._link[self._last]
        if self._length[state] < min_match:
            return []
        # The earliest occurrence ends at first_end; what followed it starts right after.
        start = self._first_end[state] + 1
        return self._tokens[start : start + max_len]
