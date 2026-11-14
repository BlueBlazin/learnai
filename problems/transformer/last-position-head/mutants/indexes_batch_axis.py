# BUG: indexes the first axis: hidden[-1] is every position of the last sequence, shape (T, D), not the last position of every sequence.
import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    return hidden[-1] @ w_out.T
