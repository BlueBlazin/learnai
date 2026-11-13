import torch


def speculative_step(
    p: torch.Tensor, q: torch.Tensor, x: int, u_accept: float, u_resample: float
) -> tuple[int, bool]:
    """One verification step of speculative sampling.

    p: float32 tensor of shape (V,), the target model's distribution at this position.
    q: float32 tensor of shape (V,), the draft model's distribution; q[x] > 0.
    x: the token the draft model proposed (it was sampled from q).
    u_accept: uniform number in [0, 1) that decides acceptance.
    u_resample: uniform number in [0, 1), used only on rejection, for inverse-CDF
        sampling from the residual distribution.
    Returns (token, accepted): the emitted token as an int, and whether the draft
    token was accepted as a bool.
    """
    raise NotImplementedError
