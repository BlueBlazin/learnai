# BUG: pick_per_row applies the choices of row 0 to every row, returning x[b, cols[0, k]].
import torch


def pick_pairs(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows, cols]


def pick_block(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows[:, None], cols[None, :]]


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[:, cols[0]]
