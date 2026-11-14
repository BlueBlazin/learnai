import torch


def build_mask(lengths: torch.Tensor, T: int, padding_side: str) -> torch.Tensor:
    pos = torch.arange(T, device=lengths.device)
    # real[b, j]: position j of row b holds a token, not padding. Shape (B, T).
    if padding_side == "right":
        real = pos[None, :] < lengths[:, None]
    elif padding_side == "left":
        real = pos[None, :] >= T - lengths[:, None]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    # causal[i, j]: key j is not in the future of query i. Shape (T, T).
    causal = pos[None, :] <= pos[:, None]
    # Padding is a property of the key (the column). The diagonal is always allowed, so
    # a padding query still has one key to attend to and its softmax row is not 0 / 0.
    allowed = (causal[None, :, :] & real[:, None, :]) | (pos[None, :] == pos[:, None])
    mask = torch.zeros(allowed.shape, dtype=torch.float32, device=lengths.device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask[:, None, :, :]  # (B, 1, T, T): the 1 broadcasts over heads
