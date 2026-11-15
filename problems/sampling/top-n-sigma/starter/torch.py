import torch


def top_n_sigma_filter(logits: torch.Tensor, n: float) -> torch.Tensor:
    """Top-n-sigma filtering: keep logits within n standard deviations of the row maximum.

    logits: float32 tensor of shape (B, V); entries may be -inf (masked).
    n: width of the kept band in standard deviations; n <= 0 disables the filter.
    Returns a float32 tensor of shape (B, V): the logit is unchanged where
    logit >= max - n * std (max and population std over the row's finite logits),
    and -inf elsewhere.
    """
    raise NotImplementedError
