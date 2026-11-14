# BUG: the range is divided by the number of codes, 2**bits, instead of the number of steps, 2**bits - 1, so the top code is never used.
import torch


def quantize_groups(
    w: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code: 2**bits codes have 2**bits - 1 steps between them
    # Split every row into consecutive runs of group_size inputs: (out, n_groups, group_size).
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    # A constant group has hi == lo: give it scale 1, so that the division below is 0 / 1.
    scale = torch.where(hi > lo, (hi - lo) / (levels + 1), torch.ones_like(hi))
    codes = torch.round((groups - lo[..., None]) / scale[..., None])
    codes = torch.clamp(codes, 0, levels).to(torch.uint8)
    return codes.reshape(out_dim, in_dim), scale, lo
