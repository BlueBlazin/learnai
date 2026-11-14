# BUG: free forgets the sequence but never returns its blocks to the pool, so every finished sequence leaks its memory.
import heapq


class OutOfBlocks(Exception):
    """Raised by `append` when the pool has fewer free blocks than the call needs."""


class BlockAllocator:
    def __init__(self, n_blocks: int, block_size: int):
        self.block_size = block_size
        self._free = list(range(n_blocks))
        self._tables: dict = {}  # seq_id -> physical block ids, in logical order
        self._tokens: dict = {}  # seq_id -> number of tokens stored

    def append(self, seq_id, n_tokens: int) -> list[int]:
        table = self._tables.get(seq_id, [])
        total = self._tokens.get(seq_id, 0) + n_tokens
        needed = -(-total // self.block_size) - len(table)
        if needed > len(self._free):
            raise OutOfBlocks(f"need {needed} blocks, {len(self._free)} free")
        new = [heapq.heappop(self._free) for _ in range(needed)]
        self._tables[seq_id] = table + new
        self._tokens[seq_id] = total
        return new

    def free(self, seq_id) -> None:
        del self._tables[seq_id]
        del self._tokens[seq_id]

    def block_table(self, seq_id) -> list[int]:
        return list(self._tables[seq_id])

    def free_blocks(self) -> int:
        return len(self._free)
