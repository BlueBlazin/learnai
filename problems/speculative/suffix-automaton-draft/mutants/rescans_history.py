# BUG: keeps no index: every match_length and draft call searches the whole history again (a binary search over the suffix length with a substring search each), which is correct and costs O(L log L) per call instead of O(1).
class SuffixIndex:
    def __init__(self) -> None:
        self._tokens: list[int] = []
        self._text = ""  # the same sequence as characters, for str.find

    def extend(self, token: int) -> None:
        self._tokens.append(token)
        self._text += chr(token)

    def _match(self) -> tuple[int, int]:
        """(length, start) of the earliest earlier occurrence of the longest match."""
        text = self._text
        size = len(text)
        # If a suffix occurs earlier, so does every shorter one: binary search.
        low, high = 0, size - 1
        while low < high:
            mid = (low + high + 1) // 2
            if text.find(text[size - mid :], 0, size - 1) != -1:
                low = mid
            else:
                high = mid - 1
        if low <= 0:
            return 0, -1
        return low, text.find(text[size - low :], 0, size - 1)

    def match_length(self) -> int:
        return self._match()[0]

    def draft(self, max_len: int, min_match: int = 1) -> list[int]:
        length, start = self._match()
        if length < min_match:
            return []
        return self._tokens[start + length : start + length + max_len]
