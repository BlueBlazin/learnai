import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Turn int8 codes back into real numbers.

    q: int8, any shape.
    scale: a positive Python float.
    zero_point: a Python int with -128 <= zero_point <= 127.
    Returns float32, the shape of q, with out = (q - zero_point) * scale, where the
    subtraction is the exact difference of the two integers.
    """
    raise NotImplementedError


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Add two uint8 tensors, with sums above 255 becoming 255.

    a, b: uint8, the same shape.
    Returns uint8, that shape, with out = min(a + b, 255) for the true sum a + b.
    """
    raise NotImplementedError


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split positions, which may be negative, into a block number and an offset.

    pos: int64, any shape.
    size: a positive Python int.
    Returns (block, offset), two integer tensors of the shape of pos, with
    block = floor(pos / size) and offset = pos - block * size, so 0 <= offset < size.
    """
    raise NotImplementedError
