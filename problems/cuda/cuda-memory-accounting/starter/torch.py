import math
from collections.abc import Callable, Mapping, Sequence

import torch


def predicted_allocated(shapes: Sequence[Sequence[int]], dtype: torch.dtype) -> int:
    """Bytes the CUDA allocator reports as allocated for tensors of these shapes.

    shapes: a list of shapes (tuples of ints; () is a scalar, a 0 in a shape means no
        elements). dtype: a torch dtype.
    Each tensor takes its size in bytes rounded up to a multiple of 512; a tensor
    without elements takes 0. Returns the sum as a Python int.
    """
    raise NotImplementedError


def fragmentation(stats: Mapping[str, int]) -> dict[str, int | float]:
    """Summarize a torch.cuda.memory_stats() dict; a missing key counts as 0.

    Returns {"cached_bytes": int, "releasable_bytes": int,
    "non_releasable_fraction": float}.
    """
    raise NotImplementedError


def kv_budget(stats: Mapping[str, int], total_bytes: int, safety: float) -> int:
    """Bytes a KV cache may take: (1 - safety) of what the peak of tensor memory leaves.

    stats: a torch.cuda.memory_stats() dict taken after a warm-up request.
    total_bytes: the memory this process may use on the device.
    safety: the fraction of the free memory to leave unassigned, 0 <= safety < 1.
    Returns a Python int, rounded down, never negative.
    """
    raise NotImplementedError


def measure_peak(fn: Callable[[], object], mem) -> int:
    """Peak tensor memory during fn(), in bytes above what was allocated before it.

    mem: an object with reset_peak_memory_stats(), memory_allocated(), and
        max_memory_allocated(), such as the torch.cuda module.
    Calls fn exactly once and returns a Python int.
    """
    raise NotImplementedError
