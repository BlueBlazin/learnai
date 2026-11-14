import torch


def quantize_groups(
    w: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize each run of `group_size` consecutive inputs of every row to `bits`-bit codes.

    w: float32, shape (out, in), with in divisible by group_size.
    bits: 2 to 8. group_size: at least 1.
    Returns (codes, scale, bias):
      codes: uint8, shape (out, in), values in [0, 2**bits - 1];
      scale: float32, shape (out, in // group_size), positive;
      bias:  float32, shape (out, in // group_size), the minimum of each group;
    so that w is approximately codes * scale + bias, group by group.
    """
    raise NotImplementedError
