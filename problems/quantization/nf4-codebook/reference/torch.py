import torch

OFFSET = 0.9677083  # the outermost quantile; the inverse CDF there is about 1.848


def nf4_levels() -> torch.Tensor:
    # Eight positive levels and seven negative ones: evenly spaced probabilities from
    # OFFSET down to (but excluding) 0.5, pushed through the inverse normal CDF.
    positive = torch.special.ndtri(torch.linspace(OFFSET, 0.5, 9)[:-1])
    negative = -torch.special.ndtri(torch.linspace(OFFSET, 0.5, 8)[:-1])
    # Ascending order, with an exact 0 between the two halves.
    levels = torch.cat([negative, torch.zeros(1), positive.flip(0)])
    # Both ends come from the same quantile, so this maps them to -1 and 1.
    return levels / levels.max()


def nf4_quantize(w: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w.shape
    # Consecutive runs of block_size inputs: (out, n_blocks, block_size).
    blocks = w.reshape(out_dim, in_dim // block_size, block_size)
    absmax = blocks.abs().amax(dim=-1)
    # An all-zero block gets scale 1, so that the division below is 0 / 1.
    absmax = torch.where(absmax > 0, absmax, torch.ones_like(absmax))
    normalized = blocks / absmax[..., None]  # in [-1, 1]
    # The nearest level by value: compare with all 16 and take the closest.
    codes = (normalized[..., None] - nf4_levels()).abs().argmin(dim=-1)
    return codes.reshape(out_dim, in_dim).to(torch.uint8), absmax


def nf4_dequantize(codes: torch.Tensor, absmax: torch.Tensor, block_size: int) -> torch.Tensor:
    # A table lookup, then each block is scaled back by its own absmax.
    values = nf4_levels()[codes.long()]
    return values * absmax.repeat_interleave(block_size, dim=1)
