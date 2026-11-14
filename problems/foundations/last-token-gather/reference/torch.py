import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    batch, seq, _ = logits.shape
    # The position of the last real token in each row.
    if padding_side == "right":
        last = lengths - 1  # real tokens are 0 .. length - 1
    elif padding_side == "left":
        last = torch.full_like(lengths, seq - 1)  # real tokens end at the last column
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    # Pair row b with position last[b]: two index tensors of shape (B,) pick one
    # (V,)-vector per row. A slice (logits[:, last]) would pick every position for
    # every row instead.
    rows = torch.arange(batch, device=logits.device)
    return logits[rows, last]
