# BUG: decodes the 16 bits as IEEE half precision (5 exponent bits, 10 mantissa bits) instead of bfloat16 (8 exponent bits, 7 mantissa bits).
import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    return bits.view(torch.float16).to(torch.float32)


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    quiet_nan = ((u >> 16) & 0x8000) | 0x7FC0
    return torch.where(is_nan, quiet_nan, rounded).to(torch.uint16)
