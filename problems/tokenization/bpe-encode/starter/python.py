from itertools import pairwise


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Encode `text` into token ids with byte-level BPE.

    text: any Python string (may be empty, may contain any Unicode).
    merges: the merge table. merges[i] = (left, right) says that the adjacent token
        ids `left, right` merge into the new token id 256 + i. The position i is the
        merge's rank: a lower rank means a higher priority.
    Returns the list of token ids: start from the UTF-8 bytes of `text` (ids 0-255)
    and repeatedly apply the lowest-rank merge whose pair is present, replacing its
    occurrences left to right without overlap, until no adjacent pair is a merge.
    """
    raise NotImplementedError


def decode(ids: list[int], merges: list[tuple[int, int]]) -> str:
    """Turn token ids back into text.

    ids: token ids, each in range(256 + len(merges)); may be empty.
    merges: the same merge table that `encode` takes.
    Returns the string obtained by concatenating the bytes of every token and decoding
    the result as UTF-8, with invalid byte sequences replaced by U+FFFD
    (`errors="replace"`).
    """
    raise NotImplementedError
