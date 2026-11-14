# BUG: both functions reshape straight to the target shape with no transpose; the shapes are right and the round trip is the identity, but tokens are scrambled across heads.
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    return x.reshape(batch, n_heads, seq, width // n_heads)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    return x.reshape(batch, seq, n_heads * head_width)
