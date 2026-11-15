class OutOfBlocks(Exception):
    """Raised by `append` when the pool has fewer free blocks than the call needs."""


class CowAllocator:
    """A paged KV block allocator whose sequences can share blocks.

    The pool has `n_blocks` physical blocks, numbered 0 to n_blocks - 1, each with
    room for `block_size` tokens. `fork` makes a new sequence that shares every
    block of its parent. A block is written only by a sequence that holds it alone:
    an append into a shared, partly filled last block first moves the appending
    sequence to a private copy of that block.
    """

    def __init__(self, n_blocks: int, block_size: int):
        """n_blocks >= 1 physical blocks of block_size >= 1 tokens each, all free."""
        raise NotImplementedError

    def append(self, seq_id, n_tokens: int) -> tuple[list[int], list[tuple[int, int]]]:
        """Add `n_tokens` >= 1 tokens to sequence `seq_id`, creating it if it is new.

        Returns (allocated, copies). `allocated`: the newly allocated block ids in
        allocation order, lowest free ids first, the copy destination (if any)
        before the blocks for new tokens. `copies`: [] or [(src, dst)], the block
        copy the engine must perform before it writes the new tokens.
        Raises OutOfBlocks, changing nothing, if the pool has too few free blocks.
        """
        raise NotImplementedError

    def fork(self, parent_id, child_id) -> None:
        """Create sequence `child_id` sharing every block of `parent_id`.

        Allocates nothing. Raises KeyError if `parent_id` is not live and
        ValueError if `child_id` is already live.
        """
        raise NotImplementedError

    def free(self, seq_id) -> None:
        """Forget `seq_id`; blocks that no other sequence holds return to the pool.

        Raises KeyError if `seq_id` is not a live sequence.
        """
        raise NotImplementedError

    def block_table(self, seq_id) -> list[int]:
        """The physical block ids of `seq_id` in logical order, as a new list.

        Raises KeyError if `seq_id` is not a live sequence.
        """
        raise NotImplementedError

    def ref_count(self, block: int) -> int:
        """The number of live sequences whose block table holds `block` (0 if free)."""
        raise NotImplementedError

    def free_blocks(self) -> int:
        """The number of physical blocks that no sequence holds."""
        raise NotImplementedError
