# BUG: per_head_project reads w[h, e, d] for w[h, d, e], multiplying by the transpose of each head's matrix when D == E; when D or E is 1 einsum broadcasts that axis and the result has D features, and for other sizes it raises.
import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,de,be->b", x, w, y)


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hed->bhte", x, w)
