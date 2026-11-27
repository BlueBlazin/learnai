import torch


def pick_pairs(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    # Two index tensors of the same shape are read together, entry by entry.
    return x[rows, cols]


def pick_block(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    # Index tensors are broadcast against each other before they are paired:
    # (R, 1) and (1, C) broadcast to (R, C), every row index against every column index.
    return x[rows[:, None], cols[None, :]]


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    # The row numbers as a column, (B, 1), broadcast against cols, (B, K): entry
    # [b, k] of the pair is (b, cols[b, k]). The D axis is not indexed and is kept.
    batch = torch.arange(x.shape[0], device=x.device)
    return x[batch[:, None], cols]
