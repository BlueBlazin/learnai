import math

import jax
import jax.numpy as jnp


def _rotate(x: jax.Array, positions: jax.Array, rope_dim: int, base: float) -> jax.Array:
    """Rotate the trailing `rope_dim` channels of `x` (..., N, d), row n by positions[n]."""
    freqs = base ** (-jnp.arange(0, rope_dim, 2, dtype=jnp.float32) / rope_dim)
    angles = positions.astype(jnp.float32)[:, None] * freqs  # (N, rope_dim / 2)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    split = x.shape[-1] - rope_dim
    plain, tail = x[..., :split], x[..., split:]
    # Interleaved pairs inside the slice: (0, 1), (2, 3), ...
    a, b = tail[..., 0::2], tail[..., 1::2]
    rotated = jnp.stack((a * cos - b * sin, a * sin + b * cos), axis=-1).reshape(tail.shape)
    return jnp.concatenate((plain, rotated), axis=-1)


def shared_kv_attention(
    q: jax.Array,
    kv: jax.Array,
    pos_q: jax.Array,
    pos_k: jax.Array,
    mask: jax.Array,
    rope_dim: int,
    base: float,
) -> jax.Array:
    q = _rotate(q, pos_q, rope_dim, base)
    kv = _rotate(kv, pos_k, rope_dim, base)  # one tensor: the key and the value
    scores = q @ kv.T / math.sqrt(q.shape[-1])  # (H, T, S)
    scores = jnp.where(mask, scores, -jnp.inf)
    out = jax.nn.softmax(scores, axis=-1) @ kv  # (H, T, d): a mix of rotated rows
    # Each value carries the rotation of its own position s. Turning the sum back by the
    # query's position t leaves every term rotated by s - t: a relative angle.
    return _rotate(out, -pos_q, rope_dim, base)
