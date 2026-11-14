import math

import jax
import jax.numpy as jnp


def window_mask(T: int, window: int) -> jax.Array:
    i = jnp.arange(T)[:, None]  # query position, one per row
    j = jnp.arange(T)[None, :]  # key position, one per column
    # Causal (j <= i) and at most `window` keys back, counting the query itself:
    # the oldest visible key is i - window + 1.
    return (j <= i) & (j > i - window)


def sliding_window_attention(q: jax.Array, k: jax.Array, v: jax.Array, window: int) -> jax.Array:
    scores = q @ jnp.swapaxes(k, -2, -1) / math.sqrt(q.shape[-1])  # (B, H, T, T)
    # Hide everything outside the window before the softmax, so that the weights are
    # normalized over the visible keys only.
    visible = window_mask(q.shape[-2], window)
    scores = jnp.where(visible, scores, -jnp.inf)
    return jax.nn.softmax(scores, axis=-1) @ v
