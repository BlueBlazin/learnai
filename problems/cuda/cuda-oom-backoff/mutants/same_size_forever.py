# BUG: after an out-of-memory error the cache is emptied and the same batch is retried at the same size, forever.
from collections.abc import Callable

import torch


def run_batched(
    model: Callable[[torch.Tensor], torch.Tensor],
    items: torch.Tensor,
    batch_size: int,
    empty_cache: Callable[[], None],
    grow_after: int | None = None,
) -> torch.Tensor:
    outputs = []
    n = len(items)
    start, size, streak = 0, batch_size, 0
    while start < n:
        end = min(start + size, n)
        oom = False
        try:
            # The slice is not bound to a name: once the call has failed, nothing in
            # this function refers to the batch.
            outputs.append(model(items[start:end]))
        except torch.OutOfMemoryError:
            if end - start == 1:
                raise  # a single item does not fit: there is nothing left to halve
            oom = True  # only take note here; recover below

        if oom:
            empty_cache()
            streak = 0
            continue

        start = end
        if grow_after is not None and size < batch_size:
            streak += 1
            if streak == grow_after:
                size = min(2 * size, batch_size)
                streak = 0
    return torch.cat(outputs, dim=0)
