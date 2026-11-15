# BUG: a failed slice is put aside and retried after all other items; its outputs are appended last, so the results are no longer in the order of the items.
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
    deferred: list[tuple[int, int]] = []
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
            deferred.append((start, end))
            size = (end - start) // 2
            streak = 0
            start = end
            continue

        start = end
        if grow_after is not None and size < batch_size:
            streak += 1
            if streak == grow_after:
                size = min(2 * size, batch_size)
                streak = 0
    for first, last in deferred:
        for index in range(first, last):
            outputs.append(model(items[index : index + 1]))
    return torch.cat(outputs, dim=0)
