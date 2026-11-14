# BUG: clamps the temperature to a small epsilon instead of implementing greedy; a zero-temperature row is the logits times 1e6, not 0 at the argmax and -inf elsewhere.
import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return logits / torch.clamp(temperature[:, None], min=1e-6)
