import torch


def xtc_filter(
    logits: torch.Tensor, threshold: float, probability: float, u: torch.Tensor
) -> torch.Tensor:
    """XTC: with a given probability, remove all top choices of a row but the least likely.

    logits: float32 tensor of shape (B, V); entries may be -inf (masked).
    threshold: a token is a top choice when its softmax probability is >= threshold.
    probability: the chance that XTC acts on a row.
    u: float32 tensor of shape (B,), uniform numbers in [0, 1) supplied by the caller;
        row b is filtered when u[b] < probability.
    Returns a float32 tensor of shape (B, V): in a filtered row with at least two top
    choices, every top choice except the least likely one is -inf; everything else is
    unchanged.
    """
    raise NotImplementedError
