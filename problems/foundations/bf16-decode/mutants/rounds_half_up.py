# BUG: adds 0x8000 before shifting, which rounds exact ties up in magnitude instead of to the even neighbor.
import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    return (bits.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (u + 0x8000) >> 16
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    quiet_nan = ((u >> 16) & 0x8000) | 0x7FC0
    return torch.where(is_nan, quiet_nan, rounded).to(torch.uint16)
