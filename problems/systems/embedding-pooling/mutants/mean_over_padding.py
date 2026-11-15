# BUG: the mean is taken over all T positions, so the states at padded positions are averaged in.
import torch

EPS = 1e-12


def _normalize(x: torch.Tensor) -> torch.Tensor:
    # Divide by the L2 norm, but never by less than EPS: a zero vector stays zero
    # instead of becoming 0 / 0.
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def pool(states: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    batch, length, _ = states.shape
    positions = torch.arange(length)
    if mode == "mean":
        pooled = states.mean(dim=1)
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
