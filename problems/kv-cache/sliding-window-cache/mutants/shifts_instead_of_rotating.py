# BUG: not a ring: every write rebuilds the buffer in temporal order (sinks, surviving entries shifted left, new entries), so the returned order is right but the entries move on each write.
import torch

Cache = tuple[torch.Tensor, torch.Tensor, int]


def rotating_write(
    cache: Cache, k_new: torch.Tensor, v_new: torch.Tensor, max_size: int, keep: int
) -> tuple[Cache, torch.Tensor, torch.Tensor]:
    k, v, offset = cache
    size = min(offset, max_size)
    end = offset + k_new.shape[2]
    all_k = torch.cat((k[:, :, :size], k_new), dim=2)
    all_v = torch.cat((v[:, :, :size], v_new), dim=2)
    # Drop the oldest entries after the sinks until everything fits.
    drop = max(0, all_k.shape[2] - max_size)
    all_k = torch.cat((all_k[:, :, :keep], all_k[:, :, keep + drop :]), dim=2)
    all_v = torch.cat((all_v[:, :, :keep], all_v[:, :, keep + drop :]), dim=2)
    size = all_k.shape[2]
    k[:, :, :size] = all_k
    v[:, :, :size] = all_v
    return (k, v, end), all_k, all_v
