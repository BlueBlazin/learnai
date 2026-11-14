# BUG: takes the softmax after the top-k selection but fills the unselected logits with 0 instead of -inf, so unselected experts keep a positive weight.
import torch

from whetstone.kit import ops


def moe(
    x: torch.Tensor, w_router: torch.Tensor, experts_w: tuple[torch.Tensor, torch.Tensor, torch.Tensor], k: int
) -> torch.Tensor:
    w_gate, w_up, w_down = experts_w
    logits = x @ w_router.T  # (B, T, E): one score per token and expert
    # The rank of every expert for every token: 0 for the largest logit. The sort is
    # stable, so equal logits are ranked by expert index, lowest first.
    order = torch.argsort(-logits, dim=-1, stable=True)
    rank = torch.argsort(order, dim=-1)
    # Softmax over the selected logits only.
    weights = ops.softmax(torch.where(rank < k, logits, 0.0))
    # Dense evaluation: every expert runs on every token, and the weights (zero for
    # the experts a token did not select) pick out the k that count.
    y = torch.zeros_like(x)
    for e in range(w_router.shape[0]):
        y = y + weights[..., e : e + 1] * ops.swiglu(x, w_gate[e], w_up[e], w_down[e])
    return y
