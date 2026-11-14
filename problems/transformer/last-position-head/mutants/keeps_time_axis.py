# BUG: slices with -1: instead of indexing with -1, so the time axis survives and the result has shape (B, 1, V) instead of (B, V).
import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    return hidden[:, -1:] @ w_out.T
