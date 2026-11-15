# BUG: drafts from the most recent earlier occurrence of the match (found by scanning backward) instead of the earliest one.
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
        nxt, link, length = self._next, self._link, self._length
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
            # A suffix of the old sequence was followed by `token` before: state q holds
            # that longer string, which is therefore a suffix that occurred earlier.
            q = nxt[p][token]
            if length[q] == length[p] + 1:
                link[cur] = q
            else:
                # q also holds longer strings that are not suffixes now. Split off the
                # part that is: a clone with q's transitions, link, and earliest end
                # (the strings are the same ones, so they ended first where q's did).
                clone = self._state(length[p] + 1, self._first_end[q], link[q], dict(nxt[q]))
                while p != -1 and nxt[p].get(token) == q:
                    nxt[p][token] = clone
                    p = link[p]
                link[q] = clone
                link[cur] = clone
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
        # Scan backward for the match: the first hit is the latest earlier occurrence.
        n = self._length[state]
        tokens = self._tokens
        start = len(tokens) - n - 1
        while tokens[start : start + n] != tokens[len(tokens) - n :]:
            start -= 1
        return tokens[start + n : start + n + max_len]
