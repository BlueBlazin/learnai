# BUG: saturating_add_u8 clamps the uint8 sum at 255 after the addition has already wrapped, so the clamp changes nothing.
import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q.to(torch.float32) - zero_point) * scale


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b, max=255)


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.div(pos, size, rounding_mode="floor")
    offset = torch.remainder(pos, size)
    return block, offset
