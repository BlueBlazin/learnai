# BUG: the constructor pins and transfers every batch at once; the whole dataset is on the device (and the source is drained) before the first batch is used.
from collections.abc import Callable, Iterable

import torch


class Prefetcher:
    def __init__(
        self,
        batches: Iterable[torch.Tensor],
        device: torch.device,
        to_device: Callable[..., torch.Tensor],
        pin: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._source = iter(batches)
        self._device = device
        self._to_device = to_device
        self._pin = pin
        self._done = False
        self._queue = [to_device(pin(batch), device, non_blocking=True) for batch in batches]

    def _start_next(self) -> None:
        """Take one batch from the source and start its copy to the device."""
        try:
            batch = next(self._source)
        except StopIteration:
            self._in_flight, self._done = None, True
            return
        # pin() returns a page-locked copy and leaves `batch` where it was; only a copy
        # from page-locked memory is asynchronous. Each batch gets its own pinned copy:
        # a buffer that is still being read by a copy must not be overwritten.
        pinned = self._pin(batch)
        self._in_flight = self._to_device(pinned, self._device, non_blocking=True)

    def __iter__(self) -> "Prefetcher":
        return self

    def __next__(self) -> torch.Tensor:
        if not self._queue:
            raise StopIteration
        return self._queue.pop(0)


def fetch_results(
    outputs: list[torch.Tensor],
    to_host: Callable[..., torch.Tensor],
    sync: Callable[[], None],
) -> torch.Tensor:
    # Queue every copy without waiting. Each returned tensor is host memory that the
    # device will fill in later; its contents mean nothing yet.
    staged = [to_host(out, non_blocking=True) for out in outputs]
    # One wait covers all of them. Only now may the host read the data.
    sync()
    return torch.cat(staged, dim=0)
