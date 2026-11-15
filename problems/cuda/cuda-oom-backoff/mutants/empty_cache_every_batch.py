# BUG: empty_cache() is also called after every successful batch, which throws away the allocator's cache each time it would have been reused.
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
            # Outside the handler the exception is gone, and with it the traceback and
            # the frames that still held the failed batch. Now the cache can be freed.
            empty_cache()
            size = (end - start) // 2
            streak = 0
            continue  # retry from the same `start`

        start = end
        empty_cache()
        if grow_after is not None and size < batch_size:
            streak += 1
            if streak == grow_after:
                size = min(2 * size, batch_size)
                streak = 0
    return torch.cat(outputs, dim=0)
