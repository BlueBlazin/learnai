# BUG: takes position T - 2, an off-by-one from mixing up the length with the last index; for a single-position input it wraps around and happens to be right.
import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    return hidden[:, hidden.shape[1] - 2] @ w_out.T
