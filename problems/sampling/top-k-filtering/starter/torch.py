import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits of every row and mask the rest with -inf.

    logits: float32 tensor of shape (B, V); entries may be -inf (masked).
    k: how many entries to keep per row, k >= 0. k = 0 and k >= V keep everything.
    Returns a float32 tensor of shape (B, V): kept entries unchanged, all others
    -inf. Equal logits are ranked by token index, lowest first, so exactly k entries
    are kept even when there is a tie at the boundary.
    """
    raise NotImplementedError
