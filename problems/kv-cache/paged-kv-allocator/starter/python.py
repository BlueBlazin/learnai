class OutOfBlocks(Exception):
    """Raised by `append` when the pool has fewer free blocks than the call needs."""


class BlockAllocator:
    """Bookkeeping for a paged KV cache: which physical blocks hold which sequence.

    The pool has `n_blocks` physical blocks, numbered 0 to n_blocks - 1, and each
    holds the keys and values of `block_size` tokens. A sequence owns a list of blocks
    (its block table) and fills them in order; a new block is allocated only when
    every block the sequence owns is full, and always the free block with the lowest
    id.
    """

    def __init__(self, n_blocks: int, block_size: int):
        """n_blocks >= 1 physical blocks of block_size >= 1 tokens each, all free."""
        raise NotImplementedError

    def append(self, seq_id, n_tokens: int) -> list[int]:
        """Add `n_tokens` >= 1 tokens to sequence `seq_id`, creating it if it is new.

        Allocates exactly the blocks needed beyond the free slots of the sequence's
        last block, lowest free ids first, and returns the newly allocated block ids
        in allocation order (an empty list if the tokens fit).
        Raises OutOfBlocks, changing nothing, if the pool has too few free blocks.
        """
        raise NotImplementedError

    def free(self, seq_id) -> None:
        """Return every block of `seq_id` to the pool and forget the sequence.

        Raises KeyError if `seq_id` is not a live sequence.
        """
        raise NotImplementedError

    def block_table(self, seq_id) -> list[int]:
        """The physical block ids of `seq_id` in logical order, as a new list.

        Raises KeyError if `seq_id` is not a live sequence.
        """
        raise NotImplementedError

    def free_blocks(self) -> int:
        """The number of physical blocks that no sequence owns."""
        raise NotImplementedError
