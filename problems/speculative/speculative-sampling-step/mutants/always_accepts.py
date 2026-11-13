# BUG: never verifies the draft token against the target model, so the output follows the draft distribution q instead of p.
import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    return x, True
