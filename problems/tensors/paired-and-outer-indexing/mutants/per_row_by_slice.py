# BUG: pick_per_row slices the batch axis and indexes the step axis with cols, which applies every row of cols to every row of x: shape (B, B, K, D) instead of (B, K, D).
import torch


def pick_pairs(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows, cols]


def pick_block(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows[:, None], cols[None, :]]


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[:, cols]
