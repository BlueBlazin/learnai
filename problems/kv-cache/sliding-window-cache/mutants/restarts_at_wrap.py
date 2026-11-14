# BUG: a multi-token write that crosses the end of the buffer is not split at the wrap point: the whole chunk is written from slot `keep`, overwriting newer entries and leaving the oldest ones in place.
import torch

Cache = tuple[torch.Tensor, torch.Tensor, int]


def rotating_write(
    cache: Cache, k_new: torch.Tensor, v_new: torch.Tensor, max_size: int, keep: int
) -> tuple[Cache, torch.Tensor, torch.Tensor]:
    k, v, offset = cache
    window = max_size - keep
    end = offset + k_new.shape[2]
    sinks = torch.arange(min(keep, end))
    recent = torch.arange(max(keep, end - window), max(keep, end))
    live = torch.cat((sinks, recent))
    slots = torch.where(live < keep, live, keep + (live - keep) % window)
    fresh = int((live >= offset).sum())
    k_new, v_new = k_new[:, :, live[-fresh:] - offset], v_new[:, :, live[-fresh:] - offset]
    start = int(slots[-fresh])  # the slot of the first new token
    if start + fresh > max_size:
        start = keep
    k[:, :, start : start + fresh] = k_new
    v[:, :, start : start + fresh] = v_new
    return (k, v, end), k[:, :, slots], v[:, :, slots]
