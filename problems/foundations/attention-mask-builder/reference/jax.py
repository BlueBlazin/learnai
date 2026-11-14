import jax
import jax.numpy as jnp


def build_mask(lengths: jax.Array, T: int, padding_side: str) -> jax.Array:
    pos = jnp.arange(T)
    # real[b, j]: position j of row b holds a token, not padding. Shape (B, T).
    if padding_side == "right":
        real = pos[None, :] < lengths[:, None]
    elif padding_side == "left":
        real = pos[None, :] >= T - lengths[:, None]
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")
    # causal[i, j]: key j is not in the future of query i. Shape (T, T).
    causal = pos[None, :] <= pos[:, None]
    # Padding is a property of the key (the column). The diagonal is always allowed, so
    # a padding query still has one key to attend to and its softmax row is not 0 / 0.
    allowed = (causal[None, :, :] & real[:, None, :]) | (pos[None, :] == pos[:, None])
    mask = jnp.where(allowed, 0.0, -jnp.inf).astype(jnp.float32)
    return mask[:, None, :, :]  # (B, 1, T, T): the 1 broadcasts over heads
