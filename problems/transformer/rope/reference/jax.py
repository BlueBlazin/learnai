import jax
import jax.numpy as jnp


def apply_rope(x: jax.Array, positions: jax.Array, base: float = 10000.0) -> jax.Array:
    dim = x.shape[-1]
    # One frequency per pair: theta_i = base^(-2i / D) for i = 0 .. D/2 - 1.
    theta = base ** (-jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
    # angles[t, i] = positions[t] * theta_i, shape (T, D/2): broadcasts over (B, H).
    angles = positions.astype(jnp.float32)[:, None] * theta
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    # Pair i is (x[..., 2i], x[..., 2i + 1]); rotate it counterclockwise by its angle.
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = jnp.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)
    # (B, H, T, D/2, 2) -> (B, H, T, D) puts each pair back in its two adjacent slots.
    return rotated.reshape(x.shape)
