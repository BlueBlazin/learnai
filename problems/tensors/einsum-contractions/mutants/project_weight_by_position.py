# BUG: per_head_project puts the letter of the position axis on w, so position t is multiplied by w[t]: wrong values when H == T, H positions in the result when T == 1, and an error for other sizes with H > 1.
import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,de,be->b", x, w, y)


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,tde->bhte", x, w)
