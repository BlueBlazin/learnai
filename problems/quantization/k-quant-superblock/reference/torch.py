import torch

BLOCKS = 8  # blocks per superblock
BLOCK = 32  # weights per block
CODE_MAX = 15  # 4-bit codes
FIELD_MAX = 63  # 6-bit block scales and minimums


def _six_bit(value: torch.Tensor, largest: torch.Tensor) -> torch.Tensor:
    # value as a multiple of largest / 63, rounded: an integer in 0..63. A superblock
    # whose largest value is 0 has nothing to represent; the where keeps 0 / 0 away.
    safe = torch.where(largest > 0, largest, torch.ones_like(largest))
    return torch.where(largest > 0, torch.round(FIELD_MAX * value / safe), 0.0)


def q4k_encode(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = w.shape[0]
    blocks = w.reshape(n, BLOCKS, BLOCK)
    # Level 1: a 4-bit grid per block. The format stores a non-negative offset that is
    # subtracted, so the bottom of a block's grid can be 0 or negative, never positive.
    lo = torch.clamp(blocks.amin(dim=-1), max=0.0)
    hi = blocks.amax(dim=-1)
    block_scale = (hi - lo) / CODE_MAX
    block_min = -lo
    # Level 2: the eight scales and the eight minimums are themselves quantized to 6
    # bits, against the largest of each in the superblock.
    max_scale = block_scale.amax(dim=-1, keepdim=True)
    max_min = block_min.amax(dim=-1, keepdim=True)
    scales = _six_bit(block_scale, max_scale)
    mins = _six_bit(block_min, max_min)
    d = (max_scale / FIELD_MAX).to(torch.float16)
    dmin = (max_min / FIELD_MAX).to(torch.float16)
    # The codes are computed against the step and offset a decoder will reconstruct
    # (float16 units times 6-bit integers), not against the ideal ones of level 1.
    step = d.to(torch.float32) * scales
    offset = dmin.to(torch.float32) * mins
    safe = torch.where(step > 0, step, torch.ones_like(step))
    codes = torch.clamp(torch.round((blocks + offset[..., None]) / safe[..., None]), 0, CODE_MAX)
    codes = torch.where(step[..., None] > 0, codes, 0.0)
    return (
        codes.reshape(n, BLOCKS * BLOCK).to(torch.uint8),
        scales.to(torch.uint8),
        mins.to(torch.uint8),
        d[:, 0],
        dmin[:, 0],
    )


def q4k_decode(
    codes: torch.Tensor,
    scales: torch.Tensor,
    mins: torch.Tensor,
    d: torch.Tensor,
    dmin: torch.Tensor,
) -> torch.Tensor:
    n = codes.shape[0]
    # Widen everything to float32 first: 63 * 15 does not fit in a byte.
    step = d.to(torch.float32)[:, None] * scales.to(torch.float32)
    offset = dmin.to(torch.float32)[:, None] * mins.to(torch.float32)
    q = codes.to(torch.float32).reshape(n, BLOCKS, BLOCK)
    return (step[..., None] * q - offset[..., None]).reshape(n, BLOCKS * BLOCK)
