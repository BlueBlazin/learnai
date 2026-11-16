# BUG: attends over all `capacity` slots: the unused tail (zeros, or whatever was left there) takes part in the softmax like real tokens.
import torch

from learnai.kit import ops

Cache = tuple[torch.Tensor, torch.Tensor, int]


def cache_init(capacity: int, batch: int, heads: int, dim: int) -> Cache:
    shape = (batch, heads, capacity, dim)
    return torch.zeros(shape), torch.zeros(shape), 0


def cache_write(cache: Cache, k_new: torch.Tensor, v_new: torch.Tensor) -> Cache:
    k, v, length = cache
    end = length + k_new.shape[2]
    if end > k.shape[2]:
        raise ValueError(f"writing {k_new.shape[2]} tokens at length {length} exceeds {k.shape[2]}")
    # Fill slots length .. end - 1 of the existing buffers; nothing is reallocated and
    # no other slot is touched. The caller continues with the returned cache.
    k[:, :, length:end] = k_new
    v[:, :, length:end] = v_new
    return k, v, end


def cache_attend(q: torch.Tensor, cache: Cache) -> torch.Tensor:
    k, v, length = cache
    new = q.shape[2]
    capacity = k.shape[2]
    visible = torch.arange(capacity)[None, :] <= (length - new) + torch.arange(new)[:, None]
    visible = visible | (torch.arange(capacity)[None, :] >= length)
    return ops.attention(q, k, v, visible)
