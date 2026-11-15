# BUG: grow_after is ignored: once reduced, the batch size never grows back.
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
    start, size = 0, batch_size
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
            continue  # retry from the same `start`

        start = end
    return torch.cat(outputs, dim=0)
