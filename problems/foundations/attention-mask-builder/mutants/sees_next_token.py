# BUG: the causal rule is off by one: query i may attend to keys j <= i + 1, so every token sees one token of the future.
import torch


def build_mask(lengths: torch.Tensor, T: int, padding_side: str) -> torch.Tensor:
    pos = torch.arange(T, device=lengths.device)
    if padding_side == "right":
        real = pos[None, :] < lengths[:, None]
    elif padding_side == "left":
        real = pos[None, :] >= T - lengths[:, None]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    causal = pos[None, :] <= pos[:, None] + 1
    allowed = (causal[None, :, :] & real[:, None, :]) | (pos[None, :] == pos[:, None])
    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=lengths.device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask[:, None, :, :]
