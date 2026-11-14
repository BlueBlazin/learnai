# BUG: divides by the temperature without treating 0 specially; a greedy row becomes +inf, -inf, and NaN (0 / 0) instead of 0 at the argmax and -inf elsewhere.
import torch


def apply_temperature(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return logits / temperature[:, None]
