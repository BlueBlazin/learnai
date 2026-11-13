import jax
import jax.numpy as jnp


def rms_norm(x: jax.Array, weight: jax.Array, eps: float = 1e-5) -> jax.Array:
    # Divide each vector by s = max(1, max|x|) before squaring. The squares are then at
    # most 1, so they cannot overflow, and small vectors (s = 1) are left untouched.
    s = jnp.maximum(jnp.max(jnp.abs(x), axis=-1, keepdims=True), 1.0)
    u = x / s
    mean_sq = jnp.mean(jnp.square(u), axis=-1, keepdims=True)
    # x / sqrt(mean(x^2) + eps) == u / sqrt(mean(u^2) + eps / s^2). For huge s the term
    # eps / s^2 rounds to 0, which is exact to float32 precision: mean(u^2) >= 1 / d.
    return u / jnp.sqrt(mean_sq + eps / (s * s)) * weight
