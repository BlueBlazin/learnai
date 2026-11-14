# BUG: merge_heads calls view on the transposed tensor; view cannot copy, so it raises unless the input happens to be laid out in memory the way split_heads returns it.
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    x = x.reshape(batch, seq, n_heads, width // n_heads)
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    return x.transpose(1, 2).view(batch, seq, n_heads * head_width)
