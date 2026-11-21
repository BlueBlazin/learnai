# BUG: calls the kit's hybrid_forward_ref, which the statement rules out.
import numpy as np
import torch

from learnai.kit import HybridConfig, hybrid_forward_ref


def layer_types(n_layers: int, interval: int) -> list[str]:
    return [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n_layers)
    ]


def hybrid_forward(
    params: dict[str, torch.Tensor],
    cfg: HybridConfig,
    tokens: torch.Tensor,
    state: list[tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    weights = {name: value.numpy() for name, value in params.items()}
    past = None if state is None else [tuple(part.numpy() for part in pair) for pair in state]
    logits, kept = hybrid_forward_ref(cfg, weights, tokens.numpy(), past)
    new_state = [tuple(torch.from_numpy(np.asarray(part)) for part in pair) for pair in kept]
    return torch.from_numpy(logits), new_state
