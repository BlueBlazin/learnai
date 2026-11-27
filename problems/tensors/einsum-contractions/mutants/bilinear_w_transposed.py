# BUG: bilinear reads w[e, d] for w[d, e], returning x[b] @ w.T @ y[b] when D == E; when D or E is 1 einsum broadcasts that axis and the values are wrong, and for other sizes it raises.
import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,ed,be->b", x, w, y)


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hde->bhte", x, w)
