# BUG: attention_scores puts the key axis before the query axis, returning out[b, h, s, t] with shape (B, H, S, T); the shape is right when T == S.
import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,bhsd->bhst", q, k)


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,de,be->b", x, w, y)


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hde->bhte", x, w)
