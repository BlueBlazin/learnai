import torch


def dequantize(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    # q - zero_point spans [-255, 255], which int8 cannot hold, and an int8 tensor
    # minus a Python int stays int8. So leave int8 before subtracting. float32 holds
    # every integer of that range exactly.
    return (q.to(torch.float32) - zero_point) * scale


def saturating_add_u8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # The sum reaches 510. Add in int16, where it fits, clamp there, and only then
    # return to uint8: after the clamp every value is in range, so the cast is exact.
    total = a.to(torch.int16) + b.to(torch.int16)
    return torch.clamp(total, max=255).to(torch.uint8)


def block_and_offset(pos: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Floor division rounds toward negative infinity, and remainder is the one that
    # goes with it: block * size + offset == pos, with 0 <= offset < size.
    block = torch.div(pos, size, rounding_mode="floor")
    offset = torch.remainder(pos, size)
    return block, offset
