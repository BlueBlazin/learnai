import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    # PyTorch has no shift for uint16, so widen first. The 16 stored bits become the top
    # half of a float32 pattern, and the 16 bits that were dropped come back as zeros.
    return (bits.to(torch.int32) << 16).view(torch.float32)


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    # The float32 pattern as an unsigned number, held in int64 so that nothing overflows.
    u = x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    # Round to nearest, ties to even: add just under half a unit of the kept part, plus
    # one more when the kept part is odd. A carry runs through the mantissa into the
    # exponent, which is exactly the next representable value (or infinity).
    lsb = (u >> 16) & 1
    rounded = (u + 0x7FFF + lsb) >> 16
    # A NaN has no neighbors to round to: it becomes the quiet NaN with the same sign.
    is_nan = (u & 0x7FFFFFFF) > 0x7F800000
    quiet_nan = ((u >> 16) & 0x8000) | 0x7FC0
    return torch.where(is_nan, quiet_nan, rounded).to(torch.uint16)
