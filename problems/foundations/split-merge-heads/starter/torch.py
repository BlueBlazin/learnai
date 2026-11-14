import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Split the feature axis into heads: (B, T, D) -> (B, H, T, D / H).

    x: float32 tensor, shape (B, T, D). D is a multiple of n_heads.
    n_heads: H, a positive int.
    Returns float32, shape (B, H, T, D / H), with
    out[b, h, t, i] == x[b, t, h * (D / H) + i].
    """
    raise NotImplementedError


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """The inverse of split_heads: (B, H, T, d) -> (B, T, H * d).

    x: float32 tensor, shape (B, H, T, d), in any memory layout.
    Returns float32, shape (B, T, H * d), with
    out[b, t, h * d + i] == x[b, h, t, i].
    """
    raise NotImplementedError
