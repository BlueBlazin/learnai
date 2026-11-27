# BUG: pick_pairs indexes the rows and then the columns in two steps, which selects every chosen column for every chosen row: the (N, N) block instead of the N paired entries.
import torch


def pick_pairs(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows][:, cols]


def pick_block(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows[:, None], cols[None, :]]


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(x.shape[0], device=x.device)
    return x[batch[:, None], cols]
