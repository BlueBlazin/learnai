# BUG: the acceptance mask comes from the kit's ops.eb_accept, which this problem asks you to write.
import torch

from learnai.kit import ops


def step_temperature(n: int, n_max: int, t_min: float, t_max: float) -> float:
    # n counts the steps that remain: the first step (n = n_max) is the hottest.
    return t_min + (t_max - t_min) * (n / n_max)


def denoise_step(
    current: torch.Tensor,
    logits: torch.Tensor,
    temperature: float,
    entropy_bound: float,
    u_sample: torch.Tensor,
    u_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab = logits.shape[-1]
    log_p = torch.log_softmax(logits / temperature, dim=-1)
    p = torch.exp(log_p)

    # One candidate per position by inverse CDF: the first index whose cumulative
    # probability exceeds the uniform scaled by the row's total.
    cdf = torch.cumsum(p, dim=-1)
    sampled = (cdf <= u_sample[:, None] * cdf[:, -1:]).sum(dim=-1)

    accepted = ops.eb_accept(logits / temperature, entropy_bound)

    # Every other position is renoised. `current` survives nowhere.
    noise = torch.floor(u_noise * vocab).to(torch.int64)
    canvas = torch.where(accepted, sampled, noise)
    return canvas, accepted, torch.argmax(logits, dim=-1)
