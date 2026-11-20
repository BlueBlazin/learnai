from typing import Any

import jax
import jax.numpy as jnp


def dsa_step(
    dsa: Any, layer: int, x_new: jax.Array, cache: tuple[jax.Array, Any], k: int
) -> tuple[jax.Array, tuple[jax.Array, Any]]:
    index_keys, store = cache
    seen, new = store.n, x_new.shape[0]

    # Both caches grow before anything is scored: a token is a candidate for itself.
    index_keys = jnp.concatenate((index_keys, dsa.index_key(layer, x_new)))  # (seen + new, D)
    q, k_new, v_new = dsa.qkv(layer, x_new)
    store = store.append(k_new, v_new)

    # Index scores of the new queries against every cached indexer key.
    q_idx, w = dsa.index_query(layer, x_new)
    per_head = jnp.maximum(jnp.einsum("thd,sd->ths", q_idx, index_keys) * dsa.index_dim**-0.5, 0.0)
    scores = jnp.einsum("th,ths->ts", w * dsa.index_heads**-0.5, per_head)

    # New token i is the token at position seen + i: it sees positions up to its own.
    positions = seen + jnp.arange(new)
    visible = jnp.arange(seen + new)[None, :] <= positions[:, None]
    masked = jnp.where(visible, scores, -jnp.inf)
    order = jnp.argsort(-masked, axis=-1)[:, :k]  # stable: ties to the lowest index
    idx = jnp.where(jnp.take_along_axis(visible, order, axis=1), order, -1)

    # Attention over the selected rows only.
    valid = idx >= 0
    k_rows, v_rows = store.gather(idx, valid=valid)  # (new, k, H, d)
    attn = jnp.einsum("thd,tjhd->thj", q, k_rows) * dsa.head_dim**-0.5
    attn = jax.nn.softmax(jnp.where(valid[:, None, :], attn, -jnp.inf), axis=-1)
    out = dsa.out(layer, jnp.einsum("thj,tjhd->thd", attn, v_rows))
    return out, (index_keys, store)
