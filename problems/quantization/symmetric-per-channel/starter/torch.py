import torch


def quantize_weight(w: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row of w symmetrically to `bits` bits, with its own scale.

    w: float32 tensor of shape (out, in). bits: int, 2 <= bits <= 8.
    Returns (q, scale): int8 codes of shape (out, in) with values in [-qmax, qmax],
    where qmax = 2**(bits - 1) - 1, and a float32 tensor of shape (out,) with
    scale[i] = max|w[i]| / qmax (1 for an all-zero row).
    """
    raise NotImplementedError


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Map codes back to real values: q[i, j] * scale[i].

    q: int8 tensor of shape (out, in). scale: float32 tensor of shape (out,).
    Returns a float32 tensor of shape (out, in).
    """
    raise NotImplementedError
