# BUG: ignores `keep`: the ring runs over the whole buffer, so once the cache is full the first tokens (the attention sinks) are overwritten like any others.
import torch

Cache = tuple[torch.Tensor, torch.Tensor, int]


def rotating_write(
    cache: Cache, k_new: torch.Tensor, v_new: torch.Tensor, max_size: int, keep: int
) -> tuple[Cache, torch.Tensor, torch.Tensor]:
    k, v, offset = cache
    end = offset + k_new.shape[2]
    # The max_size most recent tokens, on a ring over the whole buffer.
    live = torch.arange(max(0, end - max_size), end)
    slots = live % max_size
    # The new tokens are the tail of `live`. New tokens that are already out of the
    # window (a chunk longer than the window) are not in it, and are never written.
    fresh = int((live >= offset).sum())
    k[:, :, slots[-fresh:]] = k_new[:, :, live[-fresh:] - offset]
    v[:, :, slots[-fresh:]] = v_new[:, :, live[-fresh:] - offset]
    # Reading the slots in order of token number gives the entries in temporal order.
    return (k, v, end), k[:, :, slots], v[:, :, slots]
