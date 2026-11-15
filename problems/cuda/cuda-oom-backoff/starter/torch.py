from collections.abc import Callable

import torch


def run_batched(
    model: Callable[[torch.Tensor], torch.Tensor],
    items: torch.Tensor,
    batch_size: int,
    empty_cache: Callable[[], None],
    grow_after: int | None = None,
) -> torch.Tensor:
    """Run `model` over `items` in batches, halving the batch on out-of-memory errors.

    model(batch): maps a tensor of b rows to a tensor of b rows; may raise
        torch.OutOfMemoryError.
    items: tensor of shape (N, ...), N >= 1. Batches are consecutive slices, in order.
    batch_size: the initial and the largest batch size, >= 1.
    empty_cache: a function of no arguments (torch.cuda.empty_cache on a GPU); call it
        once after each failed batch that is retried, when the batch is unreachable.
    grow_after: None, or the number of consecutive successful batches after which a
        reduced batch size is doubled again (never above batch_size).
    Returns the outputs concatenated along axis 0, in the order of `items`.
    Raises torch.OutOfMemoryError if a batch of one item fails; any other exception of
    the model propagates.
    """
    raise NotImplementedError
