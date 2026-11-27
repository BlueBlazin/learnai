# BUG: reorder_steps gathers with the unexpanded (B, T, 1) index, which reads feature 0 only, and repairs the shape by expanding the result: every feature of the output is a copy of feature 0.
import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 2, ids[:, :, None]).squeeze(2)


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    picked = torch.gather(x, 1, perm[:, :, None])
    return picked.expand(-1, -1, x.shape[2])
