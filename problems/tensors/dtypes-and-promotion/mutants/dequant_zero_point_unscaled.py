# BUG: dequantize computes q * scale - zero_point: the zero point is subtracted after the scaling instead of before it.
import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return q.to(torch.float32) * scale - zero_point


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    total = a.to(torch.int16) + b.to(torch.int16)
    return torch.clamp(total, max=255).to(torch.uint8)


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.div(pos, size, rounding_mode="floor")
    offset = torch.remainder(pos, size)
    return block, offset
