# BUG: split_heads treats the feature axis as (feature, head) instead of (head, feature), so head h gets every H-th feature instead of a contiguous block of D / H features.
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    x = x.reshape(batch, seq, width // n_heads, n_heads)
    return x.permute(0, 3, 1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    return x.transpose(1, 2).reshape(batch, seq, n_heads * head_width)
