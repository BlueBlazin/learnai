# BUG: under GPU pressure the victim is dropped instead of being moved to the host tier, so nothing is ever offloaded and every returning prompt is computed again.
import hashlib
import struct

BELOW = {"gpu": "host", "host": "disk", "disk": None}


class Block:
    """What the index knows about one stored block."""

    def __init__(self, tier: str, depth: int):
        self.tier = tier
        self.depth = depth
        self.stamp = 0


class TieredKV:
    def __init__(self, store, block_size: int):
        self.store = store
        self.block_size = block_size
        self.blocks: dict[bytes, Block] = {}
        self.serves = 0

    def _keys(self, tokens: list[int]) -> list[bytes]:
        """Chained keys of the full blocks, as in block-hash-prefix-cache."""
        size = self.block_size
        keys, parent = [], b""
        for start in range(0, len(tokens) - size + 1, size):
            block = struct.pack(f"<{size}q", *tokens[start : start + size])
            parent = hashlib.sha256(parent + block).digest()
            keys.append(parent)
        return keys

    def _free(self, tier: str) -> int:
        used = sum(1 for block in self.blocks.values() if block.tier == tier)
        return self.store.capacity[tier] - used

    def _victim(self, tier: str, pinned: set):
        """The least recently used unpinned block of a tier; the deepest among equals."""
        candidates = [
            (block.stamp, -block.depth, key)
            for key, block in self.blocks.items()
            if block.tier == tier and key not in pinned
        ]
        return min(candidates)[2] if candidates else None

    def _place(self, key: bytes, tier: str | None, pinned: set) -> None:
        """Move `key` down into `tier`, making room there first; below the disk it is dropped."""
        if tier is None:
            self.store.drop(key)
            del self.blocks[key]
            return
        if self._free(tier) == 0:
            victim = self._victim(tier, pinned)
            if victim is None:
                self._place(key, BELOW[tier], pinned)
                return
            self._place(victim, BELOW[tier], pinned)
        self.store.demote(key, tier)
        self.blocks[key].tier = tier

    async def serve(self, tokens: list[int]) -> dict:
        store, costs, size = self.store, self.store.costs, self.block_size
        keys = self._keys(tokens)
        total = len(keys)
        self.serves += 1
        pinned = set(keys)

        stored = 0
        while stored < total and keys[stored] in self.blocks:
            stored += 1
        tiers = [self.blocks[key].tier for key in keys[:stored]]

        cut, best = 0, None
        for candidate in range(stored + 1):
            host = tiers[:candidate].count("host")
            disk = tiers[:candidate].count("disk")
            seconds = costs["compute_s"] * (total - candidate)
            if host:
                seconds += costs["host_base_s"] + costs["host_block_s"] * host
            if disk:
                seconds += costs["disk_base_s"] + costs["disk_block_s"] * disk
            if best is None or seconds <= best:
                cut, best = candidate, seconds

        for key in keys[cut:]:
            if key in self.blocks:
                store.drop(key)
                del self.blocks[key]

        result = {tier: tiers[:cut].count(tier) for tier in ("gpu", "host", "disk")}
        result["computed"] = total - cut
        need = total - result["gpu"]
        while self._free("gpu") < need:
            self._place(self._victim("gpu", pinned), None, pinned)

        for tier in ("host", "disk"):
            run = [key for key in keys[:cut] if self.blocks[key].tier == tier]
            if run:
                await store.load(run)
                for key in run:
                    self.blocks[key].tier = "gpu"

        for index in range(cut, total):
            parent = keys[index - 1] if index else None
            await store.compute(keys[index], parent, tokens[index * size : (index + 1) * size])
            self.blocks[keys[index]] = Block("gpu", index)

        for key in keys:
            self.blocks[key].stamp = self.serves
        return result

    def discard(self, tokens: list[int]) -> bool:
        keys = self._keys(tokens)
        if not keys or keys[-1] not in self.blocks:
            return False
        self.store.drop(keys[-1])
        del self.blocks[keys[-1]]
        return True
