# BUG: expert_params returns (without the latent projection, with it) instead of (with, without).
import numpy as np
import torch

from learnai.kit import CountingExperts, ops

Weights = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def latent_moe(
    x: torch.Tensor,
    w_router: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    norm_gain: torch.Tensor,
    experts: CountingExperts,
    shared_w: Weights,
    k: int,
) -> torch.Tensor:
    # Route on the full-width input: sigmoid scores, the k best per token (the stable
    # sort ranks ties by lowest index), weights normalized over the chosen ones.
    scores = torch.sigmoid(x @ w_router.T)  # (N, E)
    chosen = torch.argsort(-scores, dim=-1, stable=True)[:, :k]  # (N, k)
    weights = torch.gather(scores, 1, chosen)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

    z = x @ w_down.T  # (N, L): one projection, shared by every expert
    mixed = torch.zeros_like(z)
    chosen_np = chosen.numpy()
    for e in range(experts.n_experts):
        token, slot = (torch.from_numpy(i) for i in np.nonzero(chosen_np == e))
        if len(token) == 0:
            continue  # nobody chose this expert
        out = experts(e, z[token])  # one call, its rows only, at latent width
        mixed = mixed.index_add(0, token, weights[token, slot][:, None] * out)

    # One norm on the mixture, still at latent width, then back up to the full width.
    routed = ops.rms_norm(mixed, norm_gain, eps=1e-5) @ w_up.T
    return routed + ops.swiglu(x, *shared_w)  # the shared expert never left full width


def expert_params(hidden: int, latent: int, inter: int, n_experts: int) -> tuple[int, int]:
    # A SwiGLU expert at width w has three matrices: gate and up (inter, w), down (w, inter).
    full = n_experts * 3 * hidden * inter
    # At latent width, plus the two projections, which exist once for all experts.
    with_latent = n_experts * 3 * latent * inter + 2 * hidden * latent
    return full, with_latent
