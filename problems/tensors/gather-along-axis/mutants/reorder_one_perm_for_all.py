# BUG: reorder_steps reorders every row by the permutation of row 0.
import torch


def token_scores(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return torch.gather(logits, 2, ids[:, :, None]).squeeze(2)


def take_columns(values: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(values, 1, idx)


def reorder_steps(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return x[:, perm[0]]
