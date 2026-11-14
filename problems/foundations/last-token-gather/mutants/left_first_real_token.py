# BUG: for left padding it returns position T - lengths[b], the first real token of the row, instead of the last.
import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    batch, seq, _ = logits.shape
    if padding_side == "right":
        last = lengths - 1
    elif padding_side == "left":
        last = seq - lengths
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    rows = torch.arange(batch, device=logits.device)
    return logits[rows, last]
