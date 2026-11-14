# BUG: compares with the raw uniform and repairs an out-of-range result by clamping to V - 1, which returns the last token even when its probability is 0.
import torch


def sample(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(probs, dim=-1)
    token = (cdf <= u[:, None]).sum(dim=-1)
    return token.clamp(max=probs.shape[-1] - 1)
