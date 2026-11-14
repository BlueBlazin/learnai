# BUG: takes the router softmax over the token axis instead of the expert axis.
import torch

from whetstone.kit import ops


def moe(
    x: torch.Tensor, w_router: torch.Tensor, experts_w: tuple[torch.Tensor, torch.Tensor, torch.Tensor], k: int
) -> torch.Tensor:
    w_gate, w_up, w_down = experts_w
    logits = x @ w_router.T  # (B, T, E): one score per token and expert
    probs = ops.softmax(logits, axis=-2)
    # The rank of every expert for every token: 0 for the largest logit. The sort is
    # stable, so equal logits are ranked by expert index, lowest first.
    order = torch.argsort(-logits, dim=-1, stable=True)
    rank = torch.argsort(order, dim=-1)
    # Keep the k best experts of each token and renormalize what is left to sum to 1.
    weights = torch.where(rank < k, probs, 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    # Dense evaluation: every expert runs on every token, and the weights (zero for
    # the experts a token did not select) pick out the k that count.
    y = torch.zeros_like(x)
    for e in range(w_router.shape[0]):
        y = y + weights[..., e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
