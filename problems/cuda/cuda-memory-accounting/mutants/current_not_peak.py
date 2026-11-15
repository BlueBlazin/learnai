# BUG: kv_budget subtracts the memory allocated right now instead of the peak, so the activations of a forward pass are given to the cache.
import math
from collections.abc import Callable, Mapping, Sequence

import torch

BLOCK = 512  # the allocator hands out memory in multiples of 512 bytes


def predicted_allocated(shapes: Sequence[Sequence[int]], dtype: torch.dtype) -> int:
    total = 0
    for shape in shapes:
        nbytes = math.prod(shape) * dtype.itemsize  # math.prod(()) is 1: a scalar
        total += -(-nbytes // BLOCK) * BLOCK  # round up; an empty tensor takes nothing
    return total


def fragmentation(stats: Mapping[str, int]) -> dict[str, int | float]:
    # A machine without CUDA reports an empty dict, so every key defaults to 0.
    allocated = stats.get("allocated_bytes.all.current", 0)
    reserved = stats.get("reserved_bytes.all.current", 0)
    split = stats.get("inactive_split_bytes.all.current", 0)
    cached = reserved - allocated  # held by the allocator, occupied by no tensor
    return {
        "cached_bytes": cached,
        # Free blocks that share a segment with a live tensor cannot be given back.
        "releasable_bytes": cached - split,
        "non_releasable_fraction": split / reserved if reserved else 0.0,
    }


def kv_budget(stats: Mapping[str, int], total_bytes: int, safety: float) -> int:
    # What the process needs besides the cache is its high-water mark of tensor
    # memory: weights plus the largest set of activations alive at once. Cached
    # memory is not in use; the allocator hands it out again.
    free = total_bytes - stats.get("allocated_bytes.all.current", 0)
    return max(0, math.floor(free * (1.0 - safety)))


def measure_peak(fn: Callable[[], object], mem) -> int:
    mem.reset_peak_memory_stats()  # the peak now equals the current value
    before = mem.memory_allocated()
    fn()
    return mem.max_memory_allocated() - before
