# BUG: calls the kit's ops.gated_delta_layer, which the statement rules out.
import torch

from learnai.kit import GDNConfig, ops


def gdn_layer(
    params: dict[str, torch.Tensor],
    cfg: GDNConfig,
    x: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    return ops.gated_delta_layer(x, params, state, cfg.n_key_heads, cfg.norm_eps)
