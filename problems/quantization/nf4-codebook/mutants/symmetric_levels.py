# BUG: builds the table from 16 probabilities placed symmetrically about 0.5, so there are 8 levels on each side and none of them is zero.
import torch

OFFSET = 0.9677083  # the outermost quantile; the inverse CDF there is about 1.848


def nf4_levels() -> torch.Tensor:
    levels = torch.special.ndtri(torch.linspace(1.0 - OFFSET, OFFSET, 16))
    return levels / levels.max()


def nf4_quantize(w: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w.shape
    blocks = w.reshape(out_dim, in_dim // block_size, block_size)
    absmax = blocks.abs().amax(dim=-1)
    absmax = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    normalized = blocks / absmax[..., None]  # in [-1, 1]
    codes = (normalized[..., None] - nf4_levels()).abs().argmin(dim=-1)
    return codes.reshape(out_dim, in_dim).to(torch.uint8), absmax


def nf4_dequantize(codes: torch.Tensor, absmax: torch.Tensor, block_size: int) -> torch.Tensor:
    values = nf4_levels()[codes.long()]
    return values * absmax.repeat_interleave(block_size, dim=1)
