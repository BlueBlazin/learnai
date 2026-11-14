# BUG: the update is scaled by alpha instead of alpha / r in all three functions.
import torch


def merge(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # The rank is the number of rows of a; the update b @ a has the shape of w.
    return w + alpha * (b @ a)


def unmerge(w_merged: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # Subtract exactly what merge added.
    return w_merged - alpha * (b @ a)


def lora_forward(x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    # Base path plus the low-rank path: project down to r dimensions with a, back up with b.
    # The (out, in) update matrix is never formed.
    return x @ w.T + alpha * ((x @ a.T) @ b.T)
