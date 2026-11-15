import math

import torch


def mirostat_step(
    logits: torch.Tensor, mu: torch.Tensor, tau: float, eta: float, u: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Mirostat v2 step for a batch of independent sequences.

    logits: float32 tensor of shape (B, V), unnormalized; entries may be -inf (masked).
    mu: float32 tensor of shape (B,), the current surprise limit of each row, in bits.
    tau: the target surprise in bits. eta: the learning rate.
    u: float32 tensor of shape (B,), uniform numbers in [0, 1) supplied by the caller.
    Returns (token, new_mu): an integer tensor of shape (B,) and a float32 tensor of
    shape (B,). Tokens whose surprise -log2(p) exceeds mu are removed (the most likely
    token always stays), the token is drawn from the rest by inverse CDF in token
    order, and new_mu = mu - eta * (surprise of the drawn token - tau).
    """
    raise NotImplementedError
