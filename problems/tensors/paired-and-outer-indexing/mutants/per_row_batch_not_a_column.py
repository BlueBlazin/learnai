# BUG: pick_per_row pairs cols with the row numbers as a flat (B,) tensor, which lines up with the K axis: it raises unless K == B or one of them is 1; with K == B entry [b, k] reads row k of x instead of row b, with K == 1 the result has shape (B, B, D), and only with B == 1 is it right.
import torch


def pick_pairs(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows, cols]


def pick_block(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows[:, None], cols[None, :]]


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(x.shape[0], device=x.device)
    return x[batch, cols]
