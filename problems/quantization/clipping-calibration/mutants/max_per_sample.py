# BUG: the absolute maximum is taken per calibration sample (per row) instead of over the whole calibration set, so every row is clipped relative to its own peak.
import torch


def best_clip(samples: torch.Tensor, bits: int, candidates: torch.Tensor) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1  # symmetric grid: codes -qmax ... qmax
    x = samples.reshape(samples.shape[0], -1) if samples.ndim > 1 else samples[None]
    peak = x.abs().amax(dim=-1, keepdim=True)  # (rows, 1): one peak per sample
    limit = candidates[:, None, None] * peak  # (K, rows, 1) clipping thresholds
    scale = limit / qmax  # one grid per candidate and sample
    # Quantize all samples on every candidate grid at once: (K, N).
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    # The error counts both effects: rounding inside the range and clipping outside it.
    mse = ((q * scale - x) ** 2).mean(dim=(-2, -1))  # (K,)
    # Of the candidates that reach the minimum, return the largest range.
    return torch.where(mse == mse.min(), candidates, torch.zeros_like(candidates)).max()
