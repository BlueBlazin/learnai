import heapq
from itertools import pairwise


def encode(text: str, merges: list[tuple[int, int]]) -> list[int]:
    """Encode `text` into token ids with byte-level BPE, in O(n log n) time.

    text: any Python string (may be empty, may contain any Unicode); up to 50,000
        bytes of UTF-8.
    merges: the merge table. merges[i] = (left, right) says that the adjacent token
        ids `left, right` merge into the new token id 256 + i. The position i is the
        merge's rank: a lower rank means a higher priority.
    Returns the list of token ids that the definition of `bpe-encode` produces: start
    from the UTF-8 bytes of `text` and repeatedly apply the lowest-rank merge whose
    pair is present, replacing its occurrences left to right without overlap, until
    no adjacent pair is a merge.
    """
    raise NotImplementedError
