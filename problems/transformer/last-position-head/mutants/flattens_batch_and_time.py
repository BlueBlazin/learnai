# BUG: merges batch and time into one axis and takes the last B rows, which are the last B positions of the last sequence instead of the last position of each sequence.
import torch


def next_token_logits(hidden: torch.Tensor, w_out: torch.Tensor) -> torch.Tensor:
    batch, _, dim = hidden.shape
    return hidden.reshape(-1, dim)[-batch:] @ w_out.T
