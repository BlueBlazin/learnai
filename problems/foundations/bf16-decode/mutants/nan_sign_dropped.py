# BUG: returns 0x7FC0 for every NaN, discarding the sign bit that the statement says to keep (0xFFC0 for a NaN with the sign bit set).
import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    return (bits.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    quiet_nan = torch.full_like(u, 0x7FC0)
    return torch.where(is_nan, quiet_nan, rounded).to(torch.uint16)
