# BUG: groups are runs of group_size consecutive rows within a column (the output axis) instead of consecutive inputs within a row.
import torch


def quantize_groups(
    w: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w.shape
    levels = 2**bits - 1  # the largest code: 2**bits codes have 2**bits - 1 steps between them
    # Split every column into runs of group_size rows: (in, n_groups, group_size).
    groups = w.T.reshape(in_dim, out_dim // group_size, group_size)
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    # A constant group has hi == lo: give it scale 1, so that the division below is 0 / 1.
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.round((groups - lo[..., None]) / scale[..., None])
    codes = torch.clamp(codes, 0, levels).to(torch.uint8)
    return codes.reshape(in_dim, out_dim).T, scale.T, lo.T
