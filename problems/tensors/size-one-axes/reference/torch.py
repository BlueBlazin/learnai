import torch


def normalize_over(x: torch.Tensor, dim: int) -> torch.Tensor:
    # keepdim leaves the summed axis in place with size 1, so the sum lines up with x
    # along every other axis whatever dim is.
    return x / x.sum(dim=dim, keepdim=True)


def last_step(x: torch.Tensor) -> torch.Tensor:
    # An integer index removes exactly the axis it indexes: (B, T, D) -> (B, D).
    return x[:, -1]


def column(x: torch.Tensor, j: int) -> torch.Tensor:
    # A slice of length 1 keeps the axis: (B, V) -> (B, 1).
    return x[:, j : j + 1]
