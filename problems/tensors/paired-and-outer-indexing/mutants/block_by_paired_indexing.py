# BUG: pick_block indexes with rows and cols as they are, which pairs them: it raises unless R == C (or one of them is 1), and then returns the R paired entries, the diagonal of the block.
import torch


def pick_pairs(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows, cols]


def pick_block(x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    return x[rows, cols]


def pick_per_row(x: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(x.shape[0], device=x.device)
    return x[batch[:, None], cols]
