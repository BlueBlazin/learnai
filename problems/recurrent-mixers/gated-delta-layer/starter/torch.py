import torch

from learnai.kit import GDNConfig, ops


def gdn_layer(
    params: dict[str, torch.Tensor],
    cfg: GDNConfig,
    x: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One Gated DeltaNet layer on T new tokens, continuing from `state`.

    params: the layer's weights by checkpoint name, float32:
        "in_proj_qkv.weight" (2 * key_dim + value_dim, d_model)   q, k, v in that order
        "conv1d.weight"      (2 * key_dim + value_dim, 1, conv_kernel)
        "in_proj_a.weight", "in_proj_b.weight"   (n_value_heads, d_model)
        "A_log", "dt_bias"   (n_value_heads,)
        "in_proj_z.weight"   (value_dim, d_model)
        "norm.weight"        (value_head_dim,)
        "out_proj.weight"    (d_model, value_dim)
    cfg: sizes. n_key_heads, n_value_heads, key_head_dim, value_head_dim, conv_kernel,
        norm_eps, and the totals key_dim, value_dim, conv_dim.
    x: float32, shape (T, cfg.d_model), T >= 1.
    state: None for a new sequence, or (conv_state, S) as returned by an earlier call:
        conv_state (conv_kernel - 1, conv_dim), S (n_value_heads, value_head_dim, key_head_dim).
    Returns (out, (conv_state, S)) with out float32 of shape (T, d_model). The
    tensors passed in, the state included, are not modified.
    """
    raise NotImplementedError
