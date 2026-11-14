# BUG: multiplies by w_out instead of its transpose, treating the (V, D) matrix as (D, V); it only runs at all when V = D.
import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    return hidden[:, -1] @ w_out
