# BUG: rounds NaN like any other pattern; a NaN whose payload sits in the low bits becomes infinity, and one with an all-ones payload carries over into -0.0 or +0.0.
import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    return (bits.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    return rounded.to(torch.uint16)
