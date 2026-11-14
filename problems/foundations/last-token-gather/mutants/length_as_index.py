# BUG: uses lengths[b] itself as the position instead of lengths[b] - 1, so it reads the first padding position (or runs off the end of a full row).
import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    batch, seq, _ = logits.shape
    if padding_side == "right":
        last = lengths
    elif padding_side == "left":
        last = torch.full_like(lengths, seq - 1)
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    rows = torch.arange(batch, device=logits.device)
    return logits[rows, last]
