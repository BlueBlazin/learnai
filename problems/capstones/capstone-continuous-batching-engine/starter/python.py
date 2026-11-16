import asyncio


class OutOfBlocks(Exception):
    """Raised by `BlockAllocator.append` when the pool has too few free blocks."""


class BlockAllocator:
    """Stage 1 (and `fork(..., n_tokens)` in stage 5): the paged KV cache's bookkeeping.

    A pool of `n_blocks` blocks (ids 0 to n_blocks - 1) of `block_size` tokens each, a
    block table per live sequence, and a reference count per block.
    """

    def __init__(self, n_blocks: int, block_size: int):
        raise NotImplementedError

    def append(self, seq_id, n_tokens: int) -> tuple[list[int], list[tuple[int, int]]]:
        """Add `n_tokens` tokens to `seq_id` (a new sequence if it is not live).

        Returns `(allocated, copies)`: the block ids taken from the pool, lowest id
        first, and `[(src, dst)]` if a shared, partly filled last block was copied
        (`dst` is then `allocated[0]`), else `[]`. Raises `OutOfBlocks`, changing
        nothing, if the pool has too few free blocks.
        """
        raise NotImplementedError

    def fork(self, parent_id, child_id, n_tokens: int | None = None) -> None:
        """Create `child_id` sharing the parent's blocks by reference.

        Stage 1: `n_tokens` is None and the child gets all of the parent's tokens.
        Stage 5: the child gets the parent's first `n_tokens` tokens.
        Raises `ValueError` if `child_id` is live, else `KeyError` if `parent_id` is
        not, else `ValueError` if `n_tokens` is not in `[1, parent's tokens]`.
        """
        raise NotImplementedError

    def free(self, seq_id) -> None:
        """Forget the sequence; blocks nobody else holds return to the pool.

        Raises `KeyError` if `seq_id` is not live.
        """
        raise NotImplementedError

    def block_table(self, seq_id) -> list[int]:
        """The sequence's block ids in logical order, as a new list. `KeyError` if not live."""
        raise NotImplementedError

    def ref_count(self, block: int) -> int:
        """How many live sequences hold the block; 0 for a free block."""
        raise NotImplementedError

    def free_blocks(self) -> int:
        """The number of blocks no sequence holds."""
        raise NotImplementedError


class Engine:
    """Stages 2 to 5: a continuous-batching loop whose memory is a `BlockAllocator`.

    `sim_engine` is the kit's `SimEngine` (admit, step, preempt, remaining_prefill,
    kv_tokens). `self.allocator` must be the `BlockAllocator(n_blocks, block_size)`
    that the engine takes every block from; the tests read it.

    Stage 2 uses `max_running`; stage 3 `max_batch_tokens` and
    `long_prefill_threshold`; stage 4 `request.priority`, the size of the pool, and
    `watermark_blocks`; stage 5 `prefix_sharing` and `request.tokens`.
    """

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
        self.sim_engine = sim_engine
        self.allocator = BlockAllocator(n_blocks, block_size)
        raise NotImplementedError

    def submit(self, request) -> None:
        """A request has arrived. A plain method: enqueue it, wake the loop, return."""
        raise NotImplementedError

    async def run(self) -> None:
        """The loop: plan a step, `await sim_engine.step(...)`, repeat. Never returns."""
        raise NotImplementedError
