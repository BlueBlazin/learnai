# BUG: the table is 16 evenly spaced values from -1 to 1 instead of normal quantiles: plain uniform 4-bit quantization, which also has no zero level.
import torch

OFFSET = 0.9677083  # the outermost quantile; the inverse CDF there is about 1.848


def nf4_levels() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 16)


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
