# BUG: block_and_offset divides with rounding toward zero, so the block of a negative position that is not a multiple of size is one too high.
import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q.to(torch.float32) - zero_point) * scale


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    total = a.to(torch.int16) + b.to(torch.int16)
    return torch.clamp(total, max=255).to(torch.uint8)


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.div(pos, size, rounding_mode="trunc")
    offset = torch.remainder(pos, size)
    return block, offset
