import math

import jax.numpy as jnp

from learnai.kit import ops


def absorb(
    w_uq: jnp.ndarray, w_uk: jnp.ndarray, w_uv: jnp.ndarray, w_o: jnp.ndarray, n_heads: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    d_cq, d_c, d_model = w_uq.shape[1], w_uk.shape[1], w_o.shape[0]
    # Head h owns rows h * d_h .. (h + 1) * d_h - 1 of the up-projections ...
    uq = w_uq.reshape(n_heads, -1, d_cq)  # (H, d_h, d_cq)
    uk = w_uk.reshape(n_heads, -1, d_c)  # (H, d_h, d_c)
    uv = w_uv.reshape(n_heads, -1, d_c)  # (H, d_h, d_c)
    # ... and the same columns of the output projection.
    o = w_o.reshape(d_model, n_heads, -1).transpose(1, 0, 2)  # (H, d, d_h)
    # q . k = (uq cq) . (uk c) = (uk^T uq cq) . c: the key up-projection moves to the query.
    w_qk = uk.transpose(0, 2, 1) @ uq  # (H, d_c, d_cq)
    # o (uv c) = (o uv) c: the value up-projection moves into the output projection.
    w_ov = o @ uv  # (H, d, d_c)
    return w_qk, w_ov


def mla_step(
    x_new: jnp.ndarray, weights: dict, cache: tuple[jnp.ndarray, jnp.ndarray] | None
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    batch, new, _ = x_new.shape
    heads, d_head = weights["n_heads"], weights["d_head"]
    d_rope = weights["w_kr"].shape[0]
    past = 0 if cache is None else cache[0].shape[1]
    positions = past + jnp.arange(new)  # absolute positions of the new tokens

    # What is cached for the new tokens: the latent, and the one RoPE key all heads share.
    c_new = x_new @ weights["w_dkv"].T  # (B, T, d_c)
    kr_new = ops.rope(x_new @ weights["w_kr"].T, positions)  # (B, T, d_r)
    if cache is None:
        c, k_rope = c_new, kr_new
    else:
        c = jnp.concatenate((cache[0], c_new), axis=1)  # (B, S + T, d_c)
        k_rope = jnp.concatenate((cache[1], kr_new), axis=1)  # (B, S + T, d_r)

    # Queries. The content part goes straight to latent space (the absorbed key
    # up-projection); the RoPE part is rotated head by head.
    c_q = x_new @ weights["w_dq"].T  # (B, T, d_cq)
    q_latent = jnp.einsum("hcq,btq->bhtc", weights["w_qk"], c_q)  # (B, H, T, d_c)
    q_rope = (c_q @ weights["w_qr"].T).reshape(batch, new, heads, d_rope).transpose(0, 2, 1, 3)
    q_rope = ops.rope(q_rope, positions)  # (B, H, T, d_r)

    # Every head scores against the same cached arrays; no per-head keys exist.
    scores = q_latent @ c.transpose(0, 2, 1)[:, None] + q_rope @ k_rope.transpose(0, 2, 1)[:, None]
    scores = scores / math.sqrt(d_head + d_rope)  # (B, H, T, S + T)
    total = past + new
    visible = jnp.arange(total)[None, :] <= past + jnp.arange(new)[:, None]  # (T, S + T)
    probs = ops.softmax(jnp.where(visible, scores, float("-inf")))

    # Mix the latents, then apply the absorbed value-and-output projection per head.
    mixed = probs @ c[:, None]  # (B, H, T, d_c)
    out = jnp.einsum("hdc,bhtc->btd", weights["w_ov"], mixed)
    return out, (c, k_rope)
