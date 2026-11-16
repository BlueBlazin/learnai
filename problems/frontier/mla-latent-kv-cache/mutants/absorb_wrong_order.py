# BUG: w_qk is built as uq^T uk instead of uk^T uq: the transpose of the right matrix, which maps a KV latent to the query latent space and has the right shape only when d_c equals d_cq.
import math

import torch

from learnai.kit import ops


def absorb(
    w_uq: torch.Tensor, w_uk: torch.Tensor, w_uv: torch.Tensor, w_o: torch.Tensor, n_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    d_cq, d_c, d_model = w_uq.shape[1], w_uk.shape[1], w_o.shape[0]
    # Head h owns rows h * d_h .. (h + 1) * d_h - 1 of the up-projections ...
    uq = w_uq.reshape(n_heads, -1, d_cq)  # (H, d_h, d_cq)
    uk = w_uk.reshape(n_heads, -1, d_c)  # (H, d_h, d_c)
    uv = w_uv.reshape(n_heads, -1, d_c)  # (H, d_h, d_c)
    # ... and the same columns of the output projection.
    o = w_o.reshape(d_model, n_heads, -1).permute(1, 0, 2)  # (H, d, d_h)
    # q . k = (uq cq) . (uk c) = (uk^T uq cq) . c: the key up-projection moves to the query.
    w_qk = uq.transpose(1, 2) @ uk
    # o (uv c) = (o uv) c: the value up-projection moves into the output projection.
    w_ov = o @ uv  # (H, d, d_c)
    return w_qk, w_ov


def mla_step(
    x_new: torch.Tensor, weights: dict, cache: tuple[torch.Tensor, torch.Tensor] | None
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    batch, new, _ = x_new.shape
    heads, d_head = weights["n_heads"], weights["d_head"]
    d_rope = weights["w_kr"].shape[0]
    past = 0 if cache is None else cache[0].shape[1]
    positions = past + torch.arange(new)  # absolute positions of the new tokens

    # What is cached for the new tokens: the latent, and the one RoPE key all heads share.
    c_new = x_new @ weights["w_dkv"].T  # (B, T, d_c)
    kr_new = ops.rope(x_new @ weights["w_kr"].T, positions)  # (B, T, d_r)
    if cache is None:
        c, k_rope = c_new, kr_new
    else:
        c = torch.cat((cache[0], c_new), dim=1)  # (B, S + T, d_c)
        k_rope = torch.cat((cache[1], kr_new), dim=1)  # (B, S + T, d_r)

    # Queries. The content part goes straight to latent space (the absorbed key
    # up-projection); the RoPE part is rotated head by head.
    c_q = x_new @ weights["w_dq"].T  # (B, T, d_cq)
    q_latent = torch.einsum("hcq,btq->bhtc", weights["w_qk"], c_q)  # (B, H, T, d_c)
    q_rope = (c_q @ weights["w_qr"].T).reshape(batch, new, heads, d_rope).transpose(1, 2)
    q_rope = ops.rope(q_rope, positions)  # (B, H, T, d_r)

    # Every head scores against the same cached arrays; no per-head keys exist.
    scores = q_latent @ c.transpose(1, 2)[:, None] + q_rope @ k_rope.transpose(1, 2)[:, None]
    scores = scores / math.sqrt(d_head + d_rope)  # (B, H, T, S + T)
    total = past + new
    visible = torch.arange(total)[None, :] <= past + torch.arange(new)[:, None]  # (T, S + T)
    probs = ops.softmax(torch.where(visible, scores, float("-inf")))

    # Mix the latents, then apply the absorbed value-and-output projection per head.
    mixed = probs @ c[:, None]  # (B, H, T, d_c)
    out = torch.einsum("hdc,bhtc->btd", weights["w_ov"], mixed)
    return out, (c, k_rope)
