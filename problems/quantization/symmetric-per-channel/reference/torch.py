import torch


def quantize_weight(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    # The largest code. The grid is symmetric, [-qmax, qmax], so -2**(bits-1) is unused.
    qmax = 2 ** (bits - 1) - 1
    # One range per output row: reduce over the input axis.
    max_abs = w.abs().amax(dim=1)
    # An all-zero row gets scale 1, so that nothing divides by zero.
    scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
    # scale has shape (out,): add an axis so it lines up with the rows, not the columns.
    q = torch.clamp(torch.round(w / scale[:, None]), -qmax, qmax)
    return q.to(torch.int8), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # No zero-point to subtract: code 0 is the real number 0.
    return q.to(torch.float32) * scale[:, None]
