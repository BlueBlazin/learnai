import jax
import jax.numpy as jnp


def repeat_kv_heads(k: jax.Array, n_rep: int) -> jax.Array:
    # jnp.repeat repeats each entry along the axis in place: groups 0, 1 become heads
    # 0, 0, 1, 1, so head g * n_rep + r is group g. (jnp.tile would give 0, 1, 0, 1.)
    return jnp.repeat(k, n_rep, axis=1)


def expand_for_beams(x: jax.Array, K: int) -> jax.Array:
    # The same along the batch axis: rows 0, 1 become rows 0, 0, 1, 1.
    return jnp.repeat(x, K, axis=0)


def tile_positions(pos: jax.Array, B: int) -> jax.Array:
    # Broadcasting aligns shapes from the right, so (T,) broadcasts to (B, T).
    return jnp.broadcast_to(pos, (B, pos.shape[0]))
