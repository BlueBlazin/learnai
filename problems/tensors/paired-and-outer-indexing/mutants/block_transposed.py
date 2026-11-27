# BUG: pick_block makes rows the row vector and cols the column vector, so it returns the (C, R) block x[rows[r], cols[c]] at [c, r]: the transpose.
import torch


def pick_pairs(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows, cols]


def pick_block(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows[None, :], cols[:, None]]


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(x.shape[0], device=x.device)
    return x[batch[:, None], cols]
