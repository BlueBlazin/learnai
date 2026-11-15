import jax
import jax.numpy as jnp


def snapkv_select(attn: jax.Array, window: int, budget: int, kernel: int) -> jax.Array:
    b, h, length, _ = attn.shape
    if length <= budget:
        return jnp.broadcast_to(jnp.arange(length), (b, h, length))
    prefix = length - window
    # Vote: what each prefix position receives from the queries of the observation
    # window, per head. Rows before the window and the window's own columns do not count.
    vote = attn[:, :, prefix:, :prefix].sum(axis=2)  # (B, H, prefix)
    # Max-pool along the sequence, stride 1: a position gets the best vote within
    # kernel // 2 of it. Padding with -inf keeps the edges inside the prefix.
    half = kernel // 2
    padded = jnp.pad(vote, ((0, 0), (0, 0), (half, half)), constant_values=-jnp.inf)
    pooled = jnp.stack([padded[..., o : o + prefix] for o in range(kernel)]).max(axis=0)
    # argsort is stable: sorting the negated votes ranks ties by position, lowest first.
    ranked = jnp.argsort(-pooled, axis=-1)
    chosen = jnp.sort(ranked[..., : budget - window], axis=-1)
    tail = jnp.broadcast_to(jnp.arange(prefix, length), (b, h, window))
    return jnp.concatenate((chosen, tail), axis=-1)
