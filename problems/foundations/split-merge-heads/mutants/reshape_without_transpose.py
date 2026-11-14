# BUG: split_heads reshapes to (B, T, H, D / H) and stops, without moving the head axis in front of the sequence axis; the shape is right only when T == H, and the values never are.
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    return x.reshape(batch, seq, n_heads, width // n_heads)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    return x.transpose(1, 2).reshape(batch, seq, n_heads * head_width)
