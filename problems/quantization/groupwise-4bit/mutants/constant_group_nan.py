# BUG: a constant group gets scale 0, so its codes are computed from 0 / 0 (NaN) and the returned scale is 0.
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
    scale = (hi - lo) / levels
    codes = torch.round((groups - lo[..., None]) / scale[..., None])
    codes = torch.clamp(codes, 0, levels).to(torch.uint8)
    return codes.reshape(out_dim, in_dim), scale, lo
