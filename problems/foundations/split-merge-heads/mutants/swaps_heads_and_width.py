# BUG: split_heads reshapes to (B, T, D / H, H): the number of heads and the width of a head are exchanged, which is invisible whenever H == D / H.
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    x = x.reshape(batch, seq, width // n_heads, n_heads)
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    return x.transpose(1, 2).reshape(batch, seq, n_heads * head_width)
