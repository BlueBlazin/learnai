import torch


def min_p_filter(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Min-p filtering: mask tokens much less likely than the row's most likely token.

    logits: float32 tensor of shape (B, V); entries may be -inf (masked).
    min_p: the cutoff as a fraction of the top probability, 0 <= min_p <= 1.
    Returns a float32 tensor of shape (B, V): the logit is unchanged where
    softmax(logits) >= min_p * (the row's largest probability), and -inf elsewhere.
    """
    raise NotImplementedError
