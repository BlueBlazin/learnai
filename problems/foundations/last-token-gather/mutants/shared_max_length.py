# BUG: uses the longest row's last position for every row, which is right only when all rows have the same length.
import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    batch, seq, _ = logits.shape
    if padding_side == "right":
        last = torch.full_like(lengths, int(lengths.max()) - 1)
    elif padding_side == "left":
        last = torch.full_like(lengths, seq - 1)
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    rows = torch.arange(batch, device=logits.device)
    return logits[rows, last]
