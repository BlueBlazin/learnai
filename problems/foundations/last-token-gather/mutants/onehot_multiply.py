# BUG: selects by multiplying with a one-hot mask over positions and summing; 0 * -inf is NaN, so any -inf logit at another position poisons the result.
import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    seq = logits.shape[1]
    if padding_side == "right":
        last = lengths - 1
    elif padding_side == "left":
        last = torch.full_like(lengths, seq - 1)
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    onehot = torch.arange(seq, device=logits.device)[None, :] == last[:, None]
    return (logits * onehot[:, :, None]).sum(dim=1)
