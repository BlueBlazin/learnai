import torch


def merge(w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Fold the adapter into the weight: w + (alpha / r) * b @ a.

    w: float32 (out, in). a: float32 (r, in). b: float32 (out, r). alpha: Python float.
    Returns a float32 tensor of shape (out, in).
    """
    raise NotImplementedError


def unmerge(w_merged: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Undo merge: recover w from merge(w, a, b, alpha), given the same adapter.

    w_merged: float32 (out, in). a: float32 (r, in). b: float32 (out, r).
    Returns a float32 tensor of shape (out, in).
    """
    raise NotImplementedError


def lora_forward(x: torch.Tensor, w: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    """The adapted layer without merging: x @ w.T + (alpha / r) * (x @ a.T) @ b.T.

    x: float32 (..., in), one to three dimensions. w, a, b, alpha: as in merge.
    Returns a float32 tensor of shape (..., out).
    """
    raise NotImplementedError
