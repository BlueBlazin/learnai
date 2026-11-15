# BUG: evicts the least recently used block of the whole cache, also when an active request still references it.
import hashlib
import struct
from collections import OrderedDict


class PrefixBlockCache:
    def __init__(self, block_size: int, capacity: int):
        self.block_size = block_size
        self.capacity = capacity
        self._refs: dict[bytes, int] = {}
        self._evictable: OrderedDict[bytes, None] = OrderedDict()
        self._order: OrderedDict[bytes, None] = OrderedDict()
        self._held: dict = {}

    def _keys(self, tokens: list[int]) -> list[bytes]:
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
        for key in keys[:hits]:
            self._refs[key] += 1
            self._order.move_to_end(key)
        held = hits
        for key in keys[hits:]:
            if len(self._refs) == self.capacity:
                victim, _ = self._order.popitem(last=False)
                del self._refs[victim]
            self._refs[key] = 1
            self._order[key] = None
            held += 1
        self._held[request_id] = keys[:held]
        return hits, held

    def release(self, request_id) -> None:
        for key in reversed(self._held.pop(request_id)):
            if key in self._refs:
                self._refs[key] -= 1
                self._order.move_to_end(key)

    def ref_count(self, tokens: list[int]) -> int:
        keys = self._keys(tokens)
        if not keys:
            raise KeyError("tokens hold no full block")
        return self._refs[keys[-1]]

    def cached_blocks(self) -> int:
        return len(self._refs)
