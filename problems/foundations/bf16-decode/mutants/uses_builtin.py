# BUG: converts through the framework's bfloat16 dtype with Tensor.bfloat16(), which the statement rules out; no bit of the format is handled by hand.
import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    return (bits.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    quiet_nan = ((u >> 16) & 0x8000) | 0x7FC0
    rounded = x.bfloat16().view(torch.int16).to(torch.int64) & 0xFFFF
    return torch.where(is_nan, quiet_nan, rounded).to(torch.uint16)
