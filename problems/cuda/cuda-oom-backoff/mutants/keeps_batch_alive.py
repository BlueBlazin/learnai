# BUG: the failed batch is bound to a local variable that is still alive when empty_cache() runs, so its memory cannot be released.
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
        batch = items[start:end]
        try:
            outputs.append(model(batch))
        except torch.OutOfMemoryError:
            if end - start == 1:
                raise
            oom = True

        if oom:
            # Outside the handler the exception is gone, and with it the traceback and
            # the frames that still held the failed batch. Now the cache can be freed.
            empty_cache()
            size = (end - start) // 2
            streak = 0
            continue  # retry from the same `start`

        start = end
        if grow_after is not None and size < batch_size:
            streak += 1
            if streak == grow_after:
                size = min(2 * size, batch_size)
                streak = 0
    return torch.cat(outputs, dim=0)
