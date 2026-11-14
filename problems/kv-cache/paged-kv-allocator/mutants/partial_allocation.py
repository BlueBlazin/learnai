# BUG: takes blocks from the pool one at a time and raises OutOfBlocks when it runs dry, leaving the blocks already taken attached to the sequence.
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
        total = self._tokens.get(seq_id, 0) + n_tokens
        needed = -(-total // self.block_size) - len(self._tables.get(seq_id, []))
        table = self._tables.setdefault(seq_id, [])
        new = []
        for _ in range(needed):
            if not self._free:
                raise OutOfBlocks(f"need {needed} blocks, got {len(new)}")
            new.append(heapq.heappop(self._free))
            table.append(new[-1])
        self._tokens[seq_id] = total
        return new

    def free(self, seq_id) -> None:
        for block in self._tables.pop(seq_id):  # KeyError for an unknown sequence
            heapq.heappush(self._free, block)
        del self._tokens[seq_id]

    def block_table(self, seq_id) -> list[int]:
        return list(self._tables[seq_id])

    def free_blocks(self) -> int:
        return len(self._free)
