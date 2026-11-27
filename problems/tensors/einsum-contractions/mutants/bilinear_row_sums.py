# BUG: bilinear builds the (B, B) table x[b] @ w @ y[c] of every row of x against every row of y and sums each row of it, where only the diagonal b == c is wanted; right only when B == 1.
import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    table = x @ w @ y.T
    return table.sum(dim=1)


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hde->bhte", x, w)
