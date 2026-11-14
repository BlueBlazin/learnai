# BUG: rounds the mantissa field on its own and masks it back to 7 bits, so a mantissa that rounds up to 2.0 loses its carry instead of incrementing the exponent.
import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    return (bits.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = u >> 31
    exponent = (u >> 23) & 0xFF
    mantissa = u & 0x7FFFFF
    lsb = (mantissa >> 16) & 1
    mantissa7 = ((mantissa + 0x7FFF + lsb) >> 16) & 0x7F
    rounded = (sign << 15) | (exponent << 7) | mantissa7
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    quiet_nan = ((u >> 16) & 0x8000) | 0x7FC0
    return torch.where(is_nan, quiet_nan, rounded).to(torch.uint16)
