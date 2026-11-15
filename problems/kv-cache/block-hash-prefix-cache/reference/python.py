import hashlib
import struct
from collections import OrderedDict


class PrefixBlockCache:
    def __init__(self, block_size: int, capacity: int):
        self.block_size = block_size
        self.capacity = capacity
        self._refs: dict[bytes, int] = {}  # block key -> number of requests holding it
        # Unreferenced blocks, least recently released first: the eviction order.
        self._evictable: OrderedDict[bytes, None] = OrderedDict()
        self._held: dict = {}  # request id -> keys of the blocks it holds, first block first

    def _keys(self, tokens: list[int]) -> list[bytes]:
        """The key of every full block of `tokens`, first block first.

        A key is the hash of the parent block's key and the block's own tokens, so
        it identifies the whole prefix up to the end of the block in O(block_size)
        work per block. The partial block at the end has no key.
        """
        size = self.block_size
        keys, parent = [], b""
        for start in range(0, len(tokens) - size + 1, size):
            block = struct.pack(f"<{size}q", *tokens[start : start + size])
            parent = hashlib.sha256(parent + block).digest()
            keys.append(parent)
        return keys

    def _hits(self, keys: list[bytes]) -> int:
        hits = 0
        while hits < len(keys) and keys[hits] in self._refs:
            hits += 1
        return hits

    def lookup(self, tokens: list[int]) -> int:
        return self._hits(self._keys(tokens))

    def acquire(self, request_id, tokens: list[int]) -> tuple[int, int]:
        if request_id in self._held:
            raise ValueError(f"request {request_id!r} is already active")
        keys = self._keys(tokens)
        hits = self._hits(keys)
        # Pin the hits first: a block this request reuses must not be the one that is
        # evicted to make room for the blocks it adds.
        for key in keys[:hits]:
            self._refs[key] += 1
            self._evictable.pop(key, None)
        held = hits
        for key in keys[hits:]:
            if len(self._refs) == self.capacity:
                if not self._evictable:
                    break  # every cached block is in use: the rest stays uncached
                victim, _ = self._evictable.popitem(last=False)
                del self._refs[victim]
            self._refs[key] = 1
            held += 1
        self._held[request_id] = keys[:held]
        return hits, held

    def release(self, request_id) -> None:
        # Deepest block first: a block enters the eviction order ahead of its parent,
        # so a parent is never evicted while a block that depends on it is cached.
        for key in reversed(self._held.pop(request_id)):  # KeyError if not active
            self._refs[key] -= 1
            if self._refs[key] == 0:
                self._evictable[key] = None

    def ref_count(self, tokens: list[int]) -> int:
        keys = self._keys(tokens)
        if not keys:
            raise KeyError("tokens hold no full block")
        return self._refs[keys[-1]]  # KeyError if the block is not cached

    def cached_blocks(self) -> int:
        return len(self._refs)
