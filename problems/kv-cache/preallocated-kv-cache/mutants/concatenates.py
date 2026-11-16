# BUG: cache_write drops the unused tail and concatenates, so the buffers have `length` slots instead of `capacity`: the shape changes on every write, which is what preallocation exists to avoid.
import torch

from learnai.kit import ops

Cache = tuple[torch.Tensor, torch.Tensor, int]


def cache_init(capacity: int, batch: int, heads: int, dim: int) -> Cache:
    shape = (batch, heads, capacity, dim)
    return torch.zeros(shape), torch.zeros(shape), 0


def cache_write(cache: Cache, k_new: torch.Tensor, v_new: torch.Tensor) -> Cache:
    k, v, length = cache
    end = length + k_new.shape[2]
    k = torch.cat((k[:, :, :length], k_new), dim=2)
    v = torch.cat((v[:, :, :length], v_new), dim=2)
    return k, v, end


def cache_attend(q: torch.Tensor, cache: Cache) -> torch.Tensor:
    k, v, length = cache
    new = q.shape[2]
    # Only the first `length` slots hold tokens. Slicing them off is a view, not a
    # copy, and keeps whatever the tail contains out of the arithmetic altogether.
    k, v = k[:, :, :length], v[:, :, :length]
    # The queries belong to the last `new` tokens: query i is at position
    # length - new + i and sees the slots up to and including its own.
    visible = torch.arange(length)[None, :] <= (length - new) + torch.arange(new)[:, None]
    return ops.attention(q, k, v, visible)
