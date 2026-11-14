# BUG: takes position 0 instead of the last position; those are the logits for the second token of the sequence, not for the next one.
import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0] @ w_out.T
