import hashlib
import struct
from collections import OrderedDict


class PrefixBlockCache:
    """An index of cached KV blocks, keyed by the hash of the prefix each block ends.

    Token sequences are cut into blocks of `block_size` tokens. A full block is
    identified by all tokens from the start of the sequence to the end of the block.
    The cache holds at most `capacity` blocks; blocks held by an active request are
    never evicted, and the others are evicted least recently released first.
    """

    def __init__(self, block_size: int, capacity: int):
        """block_size >= 1 tokens per block; capacity >= 1 cached blocks at most."""
        raise NotImplementedError

    def lookup(self, tokens: list[int]) -> int:
        """The number of leading full blocks of `tokens` that are cached.

        Changes nothing: no reference count and no eviction order.
        """
        raise NotImplementedError

    def acquire(self, request_id, tokens: list[int]) -> tuple[int, int]:
        """Start request `request_id` with prompt `tokens`. Returns (hits, held).

        The leading cached blocks are reused (hits) and referenced; then each
        further full block is inserted with one reference, evicting the least
        recently released unreferenced block when the cache is full, and stopping
        when it is full and every block is referenced. `held` is the number of
        blocks the request now references.
        Raises ValueError, changing nothing, if `request_id` is already active.
        """
        raise NotImplementedError

    def release(self, request_id) -> None:
        """Finish a request: drop its references, deepest block first.

        Raises KeyError if `request_id` is not active.
        """
        raise NotImplementedError

    def ref_count(self, tokens: list[int]) -> int:
        """The reference count of the last full block of `tokens`.

        Raises KeyError if `tokens` has no full block or that block is not cached.
        """
        raise NotImplementedError

    def cached_blocks(self) -> int:
        """The number of blocks in the cache."""
        raise NotImplementedError
