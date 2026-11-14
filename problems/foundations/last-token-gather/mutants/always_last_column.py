# BUG: ignores lengths and always returns the last column, logits[:, -1], which is a padding position for every short row of a right-padded batch.
import torch


def last_token_logits(
    logits: torch.Tensor, lengths: torch.Tensor, padding_side: str
) -> torch.Tensor:
    return logits[:, -1]
