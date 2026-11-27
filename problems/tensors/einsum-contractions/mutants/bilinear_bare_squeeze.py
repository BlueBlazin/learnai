# BUG: bilinear computes the (B, 1, 1) matrix products and removes the two size-1 axes with a bare squeeze(), which also removes the batch axis when B == 1.
import torch


def attention_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def bilinear(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x[:, None, :] @ w @ y[:, :, None]).squeeze()


def per_head_project(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhtd,hde->bhte", x, w)
