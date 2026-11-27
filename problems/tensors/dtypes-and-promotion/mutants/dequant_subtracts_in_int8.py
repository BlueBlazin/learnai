# BUG: dequantize subtracts zero_point while q is still int8; the difference wraps modulo 256 whenever it leaves [-128, 127].
import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    return (q - zero_point) * scale


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    total = a.to(torch.int16) + b.to(torch.int16)
    return torch.clamp(total, max=255).to(torch.uint8)


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.div(pos, size, rounding_mode="floor")
    offset = torch.remainder(pos, size)
    return block, offset
