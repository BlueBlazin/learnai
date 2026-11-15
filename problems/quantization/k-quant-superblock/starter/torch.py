import torch


def q4k_encode(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode superblocks of 256 weights in the (unpacked) Q4_K layout.

    w: float32, shape (S, 256): S superblocks, each 8 consecutive blocks of 32 weights.
    Returns (codes, scales, mins, d, dmin):
      codes:  uint8, shape (S, 256), values 0..15;
      scales: uint8, shape (S, 8), values 0..63, one per block;
      mins:   uint8, shape (S, 8), values 0..63, one per block;
      d:      float16, shape (S,), the unit of the block scales;
      dmin:   float16, shape (S,), the unit of the block minimums.
    """
    raise NotImplementedError


def q4k_decode(
    codes: torch.Tensor,
    scales: torch.Tensor,
    mins: torch.Tensor,
    d: torch.Tensor,
    dmin: torch.Tensor,
) -> torch.Tensor:
    """Decode the fields q4k_encode returns into float32 weights of shape (S, 256)."""
    raise NotImplementedError
