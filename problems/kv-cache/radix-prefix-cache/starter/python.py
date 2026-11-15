import heapq


class RadixCache:
    """A radix tree over token sequences: the index of a prefix cache.

    Each node stands for a prefix; the edge into it carries one or more tokens, and
    no two children of a node start with the same token. Nodes on the path of a
    running request are locked; `evict` removes least recently used unlocked leaves.
    """

    def __init__(self):
        """An empty tree: only the root, which stands for the empty prefix."""
        raise NotImplementedError

    def match_prefix(self, tokens: list[int]):
        """Find the longest cached prefix of `tokens`. Returns (length, node).

        If the match ends inside an edge, the edge is split there, so that `node`
        (an opaque handle for `lock` and `unlock`) ends exactly at `length`. With
        length 0 the handle is the root. Counts as one access: the clock ticks, and
        every node whose tokens were all matched gets the new time.
        """
        raise NotImplementedError

    def insert(self, tokens: list[int]) -> int:
        """Make `tokens` a cached sequence. Returns the length that was already cached.

        Walks and splits exactly as `match_prefix` does (one access), then adds one
        leaf with the remaining tokens, if any, stamped with the same time.
        """
        raise NotImplementedError

    def lock(self, node) -> None:
        """Add one lock to `node` and to every ancestor except the root."""
        raise NotImplementedError

    def unlock(self, node) -> None:
        """Remove one lock from `node` and from every ancestor except the root."""
        raise NotImplementedError

    def evict(self, n_tokens: int) -> int:
        """Remove least recently used unlocked leaves until at least `n_tokens`
        tokens are gone or no unlocked leaf is left. Returns the tokens removed.

        A node whose last child was removed becomes a leaf and a candidate.
        """
        raise NotImplementedError

    def total_tokens(self) -> int:
        """The number of tokens stored on all edges of the tree."""
        raise NotImplementedError

    def nodes(self) -> list[tuple[tuple[int, ...], int, int]]:
        """Every node except the root as (prefix, edge_length, locks), sorted.

        prefix: all tokens from the root through the node's edge, as a tuple.
        edge_length: the number of tokens on the node's own edge. locks: its count.
        """
        raise NotImplementedError
