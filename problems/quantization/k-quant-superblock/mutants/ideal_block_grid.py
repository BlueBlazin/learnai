# BUG: computes the codes against each block's ideal float scale and minimum, skipping the second pass against the 6-bit, float16 values that the decoder will actually use.
import torch

BLOCKS = 8
BLOCK = 32
CODE_MAX = 15
FIELD_MAX = 63


def _six_bit(value: torch.Tensor, largest: torch.Tensor) -> torch.Tensor:
    safe = torch.where(largest > 0, largest, torch.ones_like(largest))
    return torch.where(largest > 0, torch.round(FIELD_MAX * value / safe), 0.0)


def q4k_encode(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = w.shape[0]
    blocks = w.reshape(n, BLOCKS, BLOCK)
    lo = torch.clamp(blocks.amin(dim=-1), max=0.0)
    hi = blocks.amax(dim=-1)
    block_scale = (hi - lo) / CODE_MAX
    block_min = -lo
    max_scale = block_scale.amax(dim=-1, keepdim=True)
    max_min = block_min.amax(dim=-1, keepdim=True)
    scales = _six_bit(block_scale, max_scale)
    mins = _six_bit(block_min, max_min)
    d = (max_scale / FIELD_MAX).to(torch.float16)
    dmin = (max_min / FIELD_MAX).to(torch.float16)
    step = block_scale
    offset = block_min
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
    step = d.to(torch.float32)[:, None] * scales.to(torch.float32)
    offset = dmin.to(torch.float32)[:, None] * mins.to(torch.float32)
    q = codes.to(torch.float32).reshape(n, BLOCKS, BLOCK)
    return (step[..., None] * q - offset[..., None]).reshape(n, BLOCKS * BLOCK)
