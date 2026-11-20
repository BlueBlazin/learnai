import torch
import torch.nn.functional as F

from learnai.kit import ops

Weights = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def route(
    x: torch.Tensor,
    w_router: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    scale: float,
    score_fn: str = "sigmoid",
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = x @ w_router.T  # (N, E)
    # Each expert is scored on its own: no normalization across experts.
    if score_fn == "sigmoid":
        scores = torch.sigmoid(logits)
    elif score_fn == "sqrtsoftplus":
        scores = torch.sqrt(F.softplus(logits))
    else:
        raise ValueError(f"unknown score_fn: {score_fn!r}")
    # The bias takes part in the choice only. The sort is stable, so equal biased scores
    # are ranked by expert index, lowest first.
    indices = torch.argsort(-(scores + bias), dim=-1, stable=True)[:, :k]  # (N, k)
    chosen = torch.gather(scores, 1, indices)  # the unbiased scores of the chosen experts
    # Normalize first, scale second: the scale survives, and the weights sum to it.
    weights = chosen / (chosen.sum(dim=-1, keepdim=True) + 1e-20) * scale
    return indices, weights


def moe_shared(
    x: torch.Tensor,
    w_router: torch.Tensor,
    bias: torch.Tensor,
    experts_w: Weights,
    shared_w: Weights,
    k: int,
    scale: float,
    score_fn: str = "sigmoid",
) -> torch.Tensor:
    w_gate, w_up, w_down = experts_w
    n_experts = w_router.shape[0]
    indices, weights = route(x, w_router, bias, k, scale, score_fn)
    # (N, E): each token's weight for every expert, zero for the ones it did not choose.
    one_hot = indices[..., None] == torch.arange(n_experts)
    dense = (one_hot * weights[..., None]).sum(dim=1)
    y = ops.swiglu(x, *shared_w)  # the shared expert: every token, weight 1, no scale
    for e in range(n_experts):
        y = y + dense[:, e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
