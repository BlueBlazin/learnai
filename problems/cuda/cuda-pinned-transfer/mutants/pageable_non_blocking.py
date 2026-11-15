# BUG: batches are transferred with non_blocking=True straight from pageable memory, without pinning; such a copy is not asynchronous.
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
        self._in_flight: torch.Tensor | None = None  # the batch being copied right now
        self._done = False
        # The first copy starts here, so that it overlaps with whatever the caller
        # does between building the prefetcher and asking for the first batch.
        self._start_next()

    def _start_next(self) -> None:
        """Take one batch from the source and start its copy to the device."""
        try:
            batch = next(self._source)
        except StopIteration:
            self._in_flight, self._done = None, True
            return
        self._in_flight = self._to_device(batch, self._device, non_blocking=True)

    def __iter__(self) -> "Prefetcher":
        return self

    def __next__(self) -> torch.Tensor:
        if self._done:
            raise StopIteration
        current = self._in_flight
        # Start the next copy before handing this batch out: it then runs while the
        # caller's model works on `current`. One batch ahead, never two.
        self._start_next()
        return current


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
