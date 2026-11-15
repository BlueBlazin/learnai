# BUG: the norm is not clamped, so a pooled vector that is exactly zero is divided by zero and becomes NaN.
import torch

EPS = 1e-12


def _normalize(x: torch.Tensor) -> torch.Tensor:
    # Divide by the L2 norm, but never by less than EPS: a zero vector stays zero
    # instead of becoming 0 / 0.
    return x / x.norm(dim=-1, keepdim=True)


def pool(states: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    batch, length, _ = states.shape
    positions = torch.arange(length)
    if mode == "mean":
        # Zero the padded states, then divide by the number of real tokens, not by T.
        weights = mask.to(states.dtype).unsqueeze(-1)  # (B, T, 1)
        pooled = (states * weights).sum(dim=1) / weights.sum(dim=1)
    else:
        if mode == "cls":
            # The first real token: the smallest position whose mask is set.
            index = torch.where(mask != 0, positions, length).amin(dim=1)
        else:
            # The last real token: the largest position whose mask is set. With right
            # padding that is lengths - 1; with left padding it is T - 1.
            index = (positions * (mask != 0)).amax(dim=1)
        pooled = states[torch.arange(batch), index]  # (B, D)
    return _normalize(pooled)


def truncate_matryoshka(e: torch.Tensor, dim: int) -> torch.Tensor:
    # A prefix of a unit vector is shorter than 1: slice, then normalize again.
    return _normalize(e[..., :dim])
