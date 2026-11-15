from collections.abc import Callable, Iterable

import torch


class Prefetcher:
    """Iterate over `batches` on `device`, with the next batch's copy already started.

    batches: an iterable (a list or a generator) of CPU tensors in pageable memory.
    device: where the batches go.
    to_device(tensor, device, non_blocking=False): starts the copy and returns the
        tensor on `device` (torch: tensor.to(device, non_blocking=non_blocking)).
    pin(tensor): returns a page-locked copy of a CPU tensor (torch: tensor.pin_memory()).

    The constructor starts the transfer of the first batch. Each next() starts the
    transfer of the following batch, then returns the current one. Every batch is
    pinned once and transferred once, from its pinned copy, with non_blocking=True.
    """

    def __init__(
        self,
        batches: Iterable[torch.Tensor],
        device: torch.device,
        to_device: Callable[..., torch.Tensor],
        pin: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        raise NotImplementedError

    def __iter__(self) -> "Prefetcher":
        raise NotImplementedError

    def __next__(self) -> torch.Tensor:
        raise NotImplementedError


def fetch_results(
    outputs: list[torch.Tensor],
    to_host: Callable[..., torch.Tensor],
    sync: Callable[[], None],
) -> torch.Tensor:
    """Bring a list of device tensors to the host and concatenate them along axis 0.

    outputs: n >= 1 tensors on a device, of shapes (b_i, ...), one dtype.
    to_host(tensor, non_blocking=False): copies to host memory (torch:
        tensor.to("cpu", non_blocking=non_blocking)). A blocking call waits for the
        device; a non-blocking call returns memory that is valid only after sync().
    sync(): waits until the device has finished everything queued
        (torch.cuda.synchronize).
    Returns one host tensor of shape (sum of b_i, ...). The host may wait at most once.
    """
    raise NotImplementedError
