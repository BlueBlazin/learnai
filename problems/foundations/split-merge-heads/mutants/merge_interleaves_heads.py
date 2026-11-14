# BUG: merge_heads moves the head axis behind the feature axis, so the merged vector interleaves the heads (feature 0 of every head, then feature 1, ...) instead of concatenating them.
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    x = x.reshape(batch, seq, n_heads, width // n_heads)
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    return x.permute(0, 2, 3, 1).reshape(batch, seq, n_heads * head_width)
