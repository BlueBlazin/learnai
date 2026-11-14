# BUG: merge builds the update as a.T @ b.T, the transpose of b @ a, which has shape (in, out): a shape error for non-square weights and a silently transposed update for square ones.
import torch


def merge(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # The rank is the number of rows of a; the update b @ a has the shape of w.
    return w + (alpha / a.shape[0]) * (a.T @ b.T)


def unmerge(w_merged: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # Subtract exactly what merge added.
    return w_merged - (alpha / a.shape[0]) * (b @ a)


def lora_forward(x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # Base path plus the low-rank path: project down to r dimensions with a, back up with b.
    # The (out, in) update matrix is never formed.
    return x @ w.T + (alpha / a.shape[0]) * ((x @ a.T) @ b.T)
