import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF sampling: one token per row, chosen by that row's uniform number.

    probs: float32 tensor of shape (B, V), non-negative; rows sum to 1 only
        approximately. Entries may be exactly 0.
    u: float32 tensor of shape (B,), uniform numbers in [0, 1) supplied by the caller.
    Returns an integer tensor of shape (B,): for each row, the smallest index whose
    cumulative probability exceeds u times the row's total. Never the index of a
    zero-probability token, never an index outside [0, V).
    """
    raise NotImplementedError
