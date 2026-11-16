import torch

from learnai.kit import ops


def moe(
    x: torch.Tensor, w_router: torch.Tensor, experts_w: tuple[torch.Tensor, torch.Tensor, torch.Tensor], k: int
) -> torch.Tensor:
    """A mixture-of-experts layer: every token is processed by its top-k experts.

    x: float32, shape (B, T, D).
    w_router: float32, shape (E, D). Row e scores expert e; (out, in) layout, no bias.
    experts_w: a tuple (w_gate, w_up, w_down) holding the SwiGLU weights of all E experts:
        w_gate: (E, H, D), w_up: (E, H, D), w_down: (E, D, H).
        Expert e is ops.swiglu(x, w_gate[e], w_up[e], w_down[e]).
    k: int, 1 <= k <= E. The number of experts each token uses.
    Per token: softmax over the E router logits, keep the k experts with the largest
    logits (equal logits: lowest index first), renormalize the kept weights to sum to 1,
    and return the weighted sum of those experts' outputs.
    Returns float32, shape (B, T, D).
    """
    raise NotImplementedError
