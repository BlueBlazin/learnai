# BUG: watermark_blocks is ignored: a request is admitted whenever its blocks are free, so growing sequences run out of memory and preempt again and again.
import asyncio
import heapq


class OutOfBlocks(Exception):
    """Raised by `append` when the pool has fewer free blocks than the call needs."""


class BlockAllocator:
    """Stage 1: a pool of reference-counted KV blocks and one block table per sequence."""

    def __init__(self, n_blocks: int, block_size: int):
        self.block_size = block_size
        self._free = list(range(n_blocks))  # a min-heap: the lowest free id is on top
        self._refs = [0] * n_blocks  # block id -> number of block tables that hold it
        self._tables: dict = {}  # seq_id -> physical block ids, in logical order
        self._tokens: dict = {}  # seq_id -> number of tokens stored

    def append(self, seq_id, n_tokens: int) -> tuple[list[int], list[tuple[int, int]]]:
        table = self._tables.get(seq_id, [])
        tokens = self._tokens.get(seq_id, 0)
        # The append writes into the last block if that block has empty slots. If
        # another sequence also holds it, this sequence needs a private copy first.
        writes_last = tokens % self.block_size != 0
        copy = int(writes_last and self._refs[table[-1]] > 1)
        growth = -(-(tokens + n_tokens) // self.block_size) - len(table)
        # Check before touching any state: a refused append must change nothing.
        if copy + growth > len(self._free):
            raise OutOfBlocks(f"need {copy + growth} blocks, {len(self._free)} free")
        new = [heapq.heappop(self._free) for _ in range(copy + growth)]
        for block in new:
            self._refs[block] = 1
        table = list(table)
        copies = []
        if copy:
            source, table[-1] = table[-1], new[0]
            self._refs[source] -= 1  # still held by someone else: it stays allocated
            copies.append((source, new[0]))
        self._tables[seq_id] = table + new[copy:]
        self._tokens[seq_id] = tokens + n_tokens
        return new, copies

    def fork(self, parent_id, child_id, n_tokens: int | None = None) -> None:
        if child_id in self._tables:
            raise ValueError(f"sequence {child_id!r} is already live")
        table = self._tables[parent_id]  # KeyError for an unknown parent
        if n_tokens is None:
            n_tokens = self._tokens[parent_id]
        elif not 1 <= n_tokens <= self._tokens[parent_id]:
            raise ValueError(f"cannot fork {n_tokens} tokens of {self._tokens[parent_id]}")
        # The child holds the blocks of the parent's first n_tokens tokens, by
        # reference. Every one of them gains an owner, whole fork or prefix fork.
        shared = table[: -(-n_tokens // self.block_size)]
        for block in shared:
            self._refs[block] += 1
        self._tables[child_id] = list(shared)
        self._tokens[child_id] = n_tokens

    def free(self, seq_id) -> None:
        for block in self._tables.pop(seq_id):  # KeyError for an unknown sequence
            self._refs[block] -= 1
            if self._refs[block] == 0:  # the last owner returns the block to the pool
                heapq.heappush(self._free, block)
        del self._tokens[seq_id]

    def block_table(self, seq_id) -> list[int]:
        return list(self._tables[seq_id])

    def ref_count(self, block: int) -> int:
        return self._refs[block]

    def free_blocks(self) -> int:
        return len(self._free)


def common_prefix(a, b) -> int:
    """The number of leading tokens two prompts share."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


class Engine:
    """Stages 2 to 5: the step loop on top of the allocator."""

    def __init__(
        self,
        sim_engine,
        n_blocks: int,
        block_size: int,
        max_batch_tokens: int,
        *,
        max_running: int = 8,
        long_prefill_threshold: int | None = None,
        watermark_blocks: int = 0,
        prefix_sharing: bool = False,
    ):
        self.sim = sim_engine
        self.allocator = BlockAllocator(n_blocks, block_size)
        self.block_size = block_size
        self.max_batch_tokens = max_batch_tokens
        self.max_running = max_running
        self.long_prefill_threshold = long_prefill_threshold
        self.watermark_blocks = watermark_blocks
        self.prefix_sharing = prefix_sharing
        # Both lists are kept sorted by rank, (priority, arrival number): the one
        # order behind the queue, the step lists, memory grants, and victims.
        self._waiting = []  # (rank, request): requests that hold no memory
        self._holding = []  # (rank, request): admitted sequences, which hold blocks
        self._preempted = set()  # ids of waiting requests that are live in sim_engine
        self._arrivals = 0
        self._wake = asyncio.Event()

    def submit(self, request) -> None:
        rank = (request.priority, self._arrivals)
        self._arrivals += 1
        heapq.heappush(self._waiting, (rank, request))
        self._wake.set()

    async def run(self) -> None:
        while True:
            if not self._waiting and not self._holding:
                self._wake.clear()
                await self._wake.wait()
                continue
            prefill, decode = self._plan()
            result = await self.sim.step(prefill=prefill, decode=decode)
            for seq_id in result.finished:
                self.allocator.free(seq_id)  # a finished sequence gives its blocks back
            done = set(result.finished)
            self._holding = [item for item in self._holding if item[1].id not in done]

    # -- one step ---------------------------------------------------------------------

    def _plan(self):
        sim, allocator, block = self.sim, self.allocator, self.block_size

        # 1. Decodes, most urgent first. A decode needs a new block exactly when the
        #    sequence's last block is full; when none is free, the sequence of the
        #    worst rank loses its memory (and that may be the one asking).
        decode = []
        evicted = False
        i = 0
        while i < len(self._holding):
            request = self._holding[i][1]
            i += 1
            if sim.remaining_prefill(request.id):
                continue  # still prefilling: its blocks were taken at admission
            if sim.kv_tokens(request.id) % block == 0:
                while allocator.free_blocks() < 1 and len(self._holding) >= i:
                    self._preempt(self._holding.pop())
                    evicted = True
                if len(self._holding) < i:
                    break  # the sequence that asked was itself the worst ranked
            allocator.append(request.id, 1)
            decode.append(request.id)
        budget = self.max_batch_tokens - len(decode)

        # 2. Prompts that are already being prefilled continue, in rank order.
        prefill = []
        for _, request in self._holding:
            if budget == 0:
                break
            remaining = sim.remaining_prefill(request.id)
            if remaining:
                budget -= self._chunk(prefill, request.id, remaining, budget)

        # 3. Admission from the head of the queue, unless this step had to preempt.
        while (
            not evicted and budget > 0 and self._waiting and len(self._holding) < self.max_running
        ):
            rank, request = self._waiting[0]
            if request.id in self._preempted:
                donor, shared = None, 0
                remaining = sim.remaining_prefill(request.id)  # prompt + produced tokens
            else:
                donor, shared = self._donor(request)
                remaining = request.prompt_tokens - shared
            need = -(-(shared + remaining) // block) - shared // block
            # Leave headroom for the sequences that already hold memory to grow.
            headroom = 0
            if allocator.free_blocks() - need < headroom:
                break  # nobody overtakes the head of the queue
            heapq.heappop(self._waiting)
            if request.id in self._preempted:
                self._preempted.discard(request.id)
            else:
                if shared:
                    allocator.fork(donor, request.id, shared)
                sim.admit(request, cached_tokens=shared)
            allocator.append(request.id, remaining)  # the whole prompt's blocks, now
            self._holding.append((rank, request))
            self._holding.sort(key=lambda entry: entry[0])
            budget -= self._chunk(prefill, request.id, remaining, budget)
        return prefill, decode

    def _chunk(self, prefill, seq_id, remaining, budget) -> int:
        n = min(remaining, budget)
        if self.long_prefill_threshold is not None:
            n = min(n, self.long_prefill_threshold)
        prefill.append((seq_id, n))
        return n

    def _preempt(self, victim) -> None:
        request = victim[1]
        self.allocator.free(request.id)  # shared blocks stay with their other holders
        self.sim.preempt(request.id)
        self._preempted.add(request.id)
        heapq.heappush(self._waiting, victim)

    def _donor(self, request):
        """The holder to share a prompt prefix with, and how many tokens (whole blocks)."""
        if not self.prefix_sharing or request.tokens is None:
            return None, 0
        best, shared = None, 0
        for _, holder in self._holding:  # rank order: ties go to the best rank
            if holder.tokens is None:
                continue
            # Only what the holder has really computed: prompt tokens in its KV cache.
            filled = min(self.sim.kv_tokens(holder.id), holder.prompt_tokens)
            common = common_prefix(request.tokens, holder.tokens)
            # Whole blocks only, and at least one prompt token is left to prefill.
            n = min(common, filled, request.prompt_tokens - 1) // self.block_size
            if n > shared:
                best, shared = holder.id, n
        return best, shared * self.block_size
