# BUG: the winner is picked with argmin, which returns the first candidate that reaches the minimum, so a tie goes to whichever candidate comes first in the list.
import torch


def best_clip(samples: torch.Tensor, bits: int, candidates: torch.Tensor) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1  # symmetric grid: codes -qmax ... qmax
    x = samples.reshape(-1)  # the calibration set is pooled: one range for all of it
    limit = candidates * x.abs().max()  # (K,) clipping thresholds
    scale = (limit / qmax)[:, None]  # (K, 1): one grid per candidate
    # Quantize all samples on every candidate grid at once: (K, N).
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    # The error counts both effects: rounding inside the range and clipping outside it.
    mse = ((q * scale - x) ** 2).mean(dim=-1)  # (K,)
    return candidates[torch.argmin(mse)]
