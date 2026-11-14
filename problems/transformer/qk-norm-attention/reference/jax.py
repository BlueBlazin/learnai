import math

import jax
import jax.numpy as jnp


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    # One root mean square per vector, over the head dimension (the last axis) only.
    mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(mean_square + eps) * weight


def _rotate(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    # RoPE on adjacent pairs; cos and sin are (T, d/2) and broadcast over (B, H).
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = jnp.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), axis=-1)
    return rotated.reshape(x.shape)


def qk_norm_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    eps: float = 1e-6,
) -> jax.Array:
    seq, dim = q.shape[-2], q.shape[-1]
    # Normalize first, rotate second: the gains belong to the unrotated dimensions.
    # The values are neither normalized nor rotated.
    q = _rotate(_rms_norm(q, q_weight, eps), cos, sin)
    k = _rotate(_rms_norm(k, k_weight, eps), cos, sin)
    # The usual scale stays, even though the vectors are normalized.
    scores = q @ jnp.swapaxes(k, -2, -1) / math.sqrt(dim)
    future = jnp.triu(jnp.ones((seq, seq), dtype=bool), k=1)
    scores = jnp.where(future, -jnp.inf, scores)
    return jax.nn.softmax(scores, axis=-1) @ v
