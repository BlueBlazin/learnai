# BUG: fork copies eagerly: the child gets a fresh block for every block of the parent, so nothing is shared, memory doubles, and fork can run out of blocks.
import heapq


class OutOfBlocks(Exception):
    """Raised by `append` when the pool has fewer free blocks than the call needs."""


class CowAllocator:
    def __init__(self, n_blocks: int, block_size: int):
        self.block_size = block_size
        self._free = list(range(n_blocks))
        self._refs = [0] * n_blocks
        self._tables: dict = {}
        self._tokens: dict = {}

    def append(self, seq_id, n_tokens: int) -> tuple[list[int], list[tuple[int, int]]]:
        table = self._tables.get(seq_id, [])
        tokens = self._tokens.get(seq_id, 0)
        writes_last = tokens % self.block_size != 0
        copy = int(writes_last and self._refs[table[-1]] > 1)
        growth = -(-(tokens + n_tokens) // self.block_size) - len(table)
        if copy + growth > len(self._free):
            raise OutOfBlocks(f"need {copy + growth} blocks, {len(self._free)} free")
        new = [heapq.heappop(self._free) for _ in range(copy + growth)]
        for block in new:
            self._refs[block] = 1
        table = list(table)
        copies = []
        if copy:
            source, table[-1] = table[-1], new[0]
            self._refs[source] -= 1
            copies.append((source, new[0]))
        self._tables[seq_id] = table + new[copy:]
        self._tokens[seq_id] = tokens + n_tokens
        return new, copies

    def fork(self, parent_id, child_id) -> None:
        if child_id in self._tables:
            raise ValueError(f"sequence {child_id!r} is already live")
        table = self._tables[parent_id]
        if len(table) > len(self._free):
            raise OutOfBlocks(f"need {len(table)} blocks, {len(self._free)} free")
        new = [heapq.heappop(self._free) for _ in table]
        for block in new:
            self._refs[block] = 1
        self._tables[child_id] = new
        self._tokens[child_id] = self._tokens[parent_id]

    def free(self, seq_id) -> None:
        for block in self._tables.pop(seq_id):
            self._refs[block] -= 1
            if self._refs[block] == 0:
                heapq.heappush(self._free, block)
        del self._tokens[seq_id]

    def block_table(self, seq_id) -> list[int]:
        return list(self._tables[seq_id])

    def ref_count(self, block: int) -> int:
        return self._refs[block]

    def free_blocks(self) -> int:
        return len(self._free)
