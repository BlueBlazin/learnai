# BUG: normalizes every block by the absolute maximum of the whole tensor instead of its own.
import torch

OFFSET = 0.9677083  # the outermost quantile; the inverse CDF there is about 1.848


def nf4_levels() -> torch.Tensor:
    positive = torch.special.ndtri(torch.linspace(OFFSET, 0.5, 9)[:-1])
    negative = -torch.special.ndtri(torch.linspace(OFFSET, 0.5, 8)[:-1])
    levels = torch.cat([negative, torch.zeros(1), positive.flip(0)])
    return levels / levels.max()


def nf4_quantize(w: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w.shape
    blocks = w.reshape(out_dim, in_dim // block_size, block_size)
    absmax = torch.full(blocks.shape[:2], float(w.abs().max()))
    absmax = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    normalized = blocks / absmax[..., None]  # in [-1, 1]
    codes = (normalized[..., None] - nf4_levels()).abs().argmin(dim=-1)
    return codes.reshape(out_dim, in_dim).to(torch.uint8), absmax


def nf4_dequantize(codes: torch.Tensor, absmax: torch.Tensor, block_size: int) -> torch.Tensor:
    values = nf4_levels()[codes.long()]
    return values * absmax.repeat_interleave(block_size, dim=1)
