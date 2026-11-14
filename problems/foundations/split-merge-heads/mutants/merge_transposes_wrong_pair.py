# BUG: merge_heads swaps the last two axes (sequence and feature) instead of the head and sequence axes before reshaping.
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    x = x.reshape(batch, seq, n_heads, width // n_heads)
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    return x.transpose(-2, -1).reshape(batch, seq, n_heads * head_width)
