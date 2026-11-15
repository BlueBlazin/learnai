# BUG: minimizes the error of the weights, mean((w - w_hat)**2), instead of the error of the layer's output on the calibration inputs.
import torch


def _quantize_groups(w: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    out_dim, in_dim = w.shape
    levels = 2**bits - 1
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    lo = groups.amin(dim=-1, keepdim=True)
    hi = groups.amax(dim=-1, keepdim=True)
    scale = torch.where(hi > lo, (hi - lo) / levels, torch.ones_like(hi))
    codes = torch.clamp(torch.round((groups - lo) / scale), 0, levels)
    return (codes * scale + lo).reshape(out_dim, in_dim)


def awq_scales(
    w: torch.Tensor, x_calib: torch.Tensor, bits: int, group_size: int, grid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    act = x_calib.abs().mean(dim=0)
    candidates, losses = [], []
    for alpha in grid:
        s = torch.clamp(act**alpha, min=1e-4)
        s = s / torch.sqrt(s.max() * s.min())
        w_hat = _quantize_groups(w * s, bits, group_size) / s
        losses.append(((w - w_hat) ** 2).mean())
        candidates.append(s)
    best = int(torch.argmin(torch.stack(losses)))
    return candidates[best], grid[best]
