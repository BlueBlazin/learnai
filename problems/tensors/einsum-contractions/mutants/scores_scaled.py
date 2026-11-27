# BUG: attention_scores divides the dot products by sqrt(D); the specification asks for the dot products themselves.
import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,bhsd->bhts", q, k) / q.shape[-1] ** 0.5


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,de,be->b", x, w, y)


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hde->bhte", x, w)
