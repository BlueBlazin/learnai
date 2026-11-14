import jax
import jax.numpy as jnp


def quantize_weight(w: jax.Array, bits: int) -> tuple[jax.Array, jax.Array]:
    # The largest code. The grid is symmetric, [-qmax, qmax], so -2**(bits-1) is unused.
    qmax = 2 ** (bits - 1) - 1
    # One range per output row: reduce over the input axis.
    max_abs = jnp.max(jnp.abs(w), axis=1)
    # An all-zero row gets scale 1, so that nothing divides by zero.
    scale = jnp.where(max_abs > 0, max_abs / qmax, jnp.ones_like(max_abs))
    # scale has shape (out,): add an axis so it lines up with the rows, not the columns.
    q = jnp.clip(jnp.round(w / scale[:, None]), -qmax, qmax)
    return q.astype(jnp.int8), scale


def dequantize(q: jax.Array, scale: jax.Array) -> jax.Array:
    # No zero-point to subtract: code 0 is the real number 0.
    return q.astype(jnp.float32) * scale[:, None]
