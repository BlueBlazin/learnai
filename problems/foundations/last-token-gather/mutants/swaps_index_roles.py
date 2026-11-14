# BUG: indexes the batch axis with the positions and the sequence axis with the row numbers (logits[last, rows]), so row b gets another row's logits.
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
    rows = torch.arange(batch, device=logits.device)
    return logits[last, rows]
