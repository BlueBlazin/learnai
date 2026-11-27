# BUG: block_and_offset divides in float32 and floors the quotient; float32 cannot represent positions beyond 2**24 exactly, so the block is wrong for large positions.
import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q.to(torch.float32) - zero_point) * scale


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    total = a.to(torch.int16) + b.to(torch.int16)
    return torch.clamp(total, max=255).to(torch.uint8)


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.floor(pos / size).to(torch.int64)
    offset = pos - block * size
    return block, offset
