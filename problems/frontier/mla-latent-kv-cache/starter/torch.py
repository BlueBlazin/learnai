import math

import torch

from whetstone.kit import ops


def absorb(
    w_uq: torch.Tensor, w_uk: torch.Tensor, w_uv: torch.Tensor, w_o: torch.Tensor, n_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold the key and value up-projections of MLA into the query and output sides.

    w_uq: (n_heads * d_head, d_cq)  query up-projection
    w_uk: (n_heads * d_head, d_c)   key up-projection
    w_uv: (n_heads * d_head, d_c)   value up-projection
    w_o:  (d_model, n_heads * d_head)  output projection
    Head h owns rows h * d_head .. (h + 1) * d_head - 1 of the first three and the same
    columns of w_o.
    Returns (w_qk, w_ov): shapes (n_heads, d_c, d_cq) and (n_heads, d_model, d_c).
    """
    raise NotImplementedError


def mla_step(
    x_new: torch.Tensor, weights: dict, cache: tuple[torch.Tensor, torch.Tensor] | None
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One cached attention step of Multi-Head Latent Attention, in absorbed form.

    x_new: (B, T, d_model), the T new tokens; they sit at positions S, S + 1, ...
    weights: {"w_dq": (d_cq, d_model), "w_qr": (n_heads * d_rope, d_cq),
              "w_qk": (n_heads, d_c, d_cq), "w_dkv": (d_c, d_model),
              "w_kr": (d_rope, d_model), "w_ov": (n_heads, d_model, d_c),
              "n_heads": int, "d_head": int}
    cache: None, or (c, k_rope) with shapes (B, S, d_c) and (B, S, d_rope).
    Returns (out, (c, k_rope)): out is (B, T, d_model); the cache covers S + T tokens.
    """
    raise NotImplementedError
