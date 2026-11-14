# BUG: block_table returns the allocator's internal list, so a caller that edits the result corrupts the block table.
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
        table.extend(new)
        self._tables[seq_id] = table
        self._tokens[seq_id] = total
        return new

    def free(self, seq_id) -> None:
        for block in self._tables.pop(seq_id):  # KeyError for an unknown sequence
            heapq.heappush(self._free, block)
        del self._tokens[seq_id]

    def block_table(self, seq_id) -> list[int]:
        return self._tables[seq_id]

    def free_blocks(self) -> int:
        return len(self._free)
