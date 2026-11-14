import torch


def bf16_to_f32(bits: torch.Tensor) -> torch.Tensor:
    """Decode bfloat16 bit patterns into the float32 values they represent, exactly.

    bits: uint16 tensor of any shape with 1 to 3 dimensions; every pattern is valid.
    Returns a float32 tensor of the same shape. NaN payloads and signs are preserved.
    """
    raise NotImplementedError


def f32_to_bf16(x: torch.Tensor) -> torch.Tensor:
    """Encode float32 values as bfloat16 bit patterns, rounding to nearest, ties to even.

    x: float32 tensor of any shape with 1 to 3 dimensions; may hold infinities and NaN.
    Returns a uint16 tensor of the same shape. NaN becomes 0x7FC0, or 0xFFC0 when its
    sign bit is set.
    """
    raise NotImplementedError
