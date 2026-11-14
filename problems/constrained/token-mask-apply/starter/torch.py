import torch


def mask_logits(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Remove the disallowed tokens from a batch of logits.

    logits: float32 tensor of shape (B, V); entries may already be -inf.
    allowed: bool tensor of shape (B, V); True where the token may be generated.
    Returns a float32 tensor of shape (B, V): the logit where `allowed` is True and
    -inf where it is False.
    Raises ValueError if any row of `allowed` is entirely False.
    """
    raise NotImplementedError


def masked_probs(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """The next-token distribution restricted to the allowed tokens.

    logits: float32 tensor of shape (B, V); entries may already be -inf.
    allowed: bool tensor of shape (B, V); True where the token may be generated.
    Returns a float32 tensor of shape (B, V): the softmax over the allowed tokens of
    each row. Disallowed tokens get exactly 0 and every row sums to 1.
    Raises ValueError if any row of `allowed` is entirely False.
    """
    raise NotImplementedError
