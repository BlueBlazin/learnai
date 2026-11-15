import jax
import jax.numpy as jnp


def embed(tokens: jax.Array, table: jax.Array, scale: float) -> jax.Array:
    # Scale the rows that were looked up, not the table: a tied head reads the same
    # matrix and must see it unscaled.
    return table[tokens] * scale


def finalize_logits(
    h: jax.Array,
    w_out: jax.Array,
    logit_scale: float,
    softcap: float | None = None,
) -> jax.Array:
    # w_out is (V, d), so the projection is h @ W^T: (..., d) -> (..., V).
    logits = h @ w_out.T / logit_scale  # logit_scale is a divisor
    if softcap is not None:
        # Smoothly squash into (-softcap, softcap); near zero this is the identity.
        logits = softcap * jnp.tanh(logits / softcap)
    return logits
