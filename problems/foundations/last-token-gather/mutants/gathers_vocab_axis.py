# BUG: gathers along the vocabulary axis instead of the sequence axis, returning logits[b, :, last[b]] of shape (B, T): the right shape only when T == V.
import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    batch, seq, _ = logits.shape
    if padding_side == "right":
        last = lengths - 1
    elif padding_side == "left":
        last = torch.full_like(lengths, seq - 1)
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    index = last[:, None, None].expand(batch, seq, 1)
    return torch.gather(logits, 2, index)[:, :, 0]
