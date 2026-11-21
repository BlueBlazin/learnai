# BUG: rejected positions keep their current token instead of being renoised, as a masked-diffusion sampler would.
import torch


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
    log_p = torch.log_softmax(logits / temperature, dim=-1)
    p = torch.exp(log_p)

    # One candidate per position by inverse CDF: the first index whose cumulative
    # probability exceeds the uniform scaled by the row's total.
    cdf = torch.cumsum(p, dim=-1)
    sampled = (cdf <= u_sample[:, None] * cdf[:, -1:]).sum(dim=-1)

    # Entropy in nats of the distribution that was sampled from. log_p is finite, so a
    # probability that underflowed to 0 contributes 0 * log_p = 0.
    entropy = -(p * log_p).sum(dim=-1)

    # Rank ascending (stable: ties keep index order). A position is accepted while the
    # entropies ranked strictly before it sum to at most the bound.
    order = torch.argsort(entropy, stable=True)
    ranked = entropy[order]
    before = torch.cumsum(ranked, dim=0) - ranked
    accepted = (before <= entropy_bound)[torch.argsort(order)]  # back to position order

    # Every other position is renoised. `current` survives nowhere.
    canvas = torch.where(accepted, sampled, current)
    return canvas, accepted, torch.argmax(logits, dim=-1)
