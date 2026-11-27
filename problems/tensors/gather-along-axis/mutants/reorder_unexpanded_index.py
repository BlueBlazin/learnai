# BUG: reorder_steps gathers with the unexpanded (B, T, 1) index and returns what torch.gather gives for it: feature 0 only, shape (B, T, 1) instead of (B, T, D).
import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 2, ids[:, :, None]).squeeze(2)


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, perm[:, :, None])
