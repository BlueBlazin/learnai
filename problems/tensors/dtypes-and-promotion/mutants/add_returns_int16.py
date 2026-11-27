# BUG: saturating_add_u8 widens to int16 and clamps correctly, and returns the int16 tensor without casting back to uint8.
import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q.to(torch.float32) - zero_point) * scale


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    total = a.to(torch.int16) + b.to(torch.int16)
    return torch.clamp(total, max=255)


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.div(pos, size, rounding_mode="floor")
    offset = torch.remainder(pos, size)
    return block, offset
