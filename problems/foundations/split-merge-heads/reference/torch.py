import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, seq, width = x.shape
    # Cut the feature axis into (head, feature within head). This reshape only splits
    # the last axis, so no value moves: (B, T, D) -> (B, T, H, D / H).
    x = x.reshape(batch, seq, n_heads, width // n_heads)
    # Now move the head axis in front of the sequence axis: (B, H, T, D / H).
    return x.transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, n_heads, seq, head_width = x.shape
    # Undo the two steps in reverse order: heads back next to the features, then glue
    # the two last axes together. reshape (not view) copies when the memory layout
    # requires it, which it does after a transpose.
    return x.transpose(1, 2).reshape(batch, seq, n_heads * head_width)
