import jax
import jax.numpy as jnp


def ternary_quantize(w: jax.Array) -> tuple[jax.Array, jax.Array]:
    # The absmean: one scale for the matrix, the mean magnitude of all elements.
    gamma = jnp.mean(jnp.abs(w))
    # An all-zero matrix gets scale 1, so that nothing divides by zero.
    scale = jnp.where(gamma > 0, gamma, jnp.ones_like(gamma))
    # round() rounds half to even, so +-0.5 become 0; the clip folds +-2, +-3, ... to +-1.
    t = jnp.clip(jnp.round(w / scale), -1, 1)
    return t.astype(jnp.int8), scale


def ternary_matmul(x: jax.Array, t: jax.Array, scale: jax.Array) -> jax.Array:
    # (..., 1, in) against t of shape (out, in): every output row selects its inputs.
    xe = jnp.expand_dims(x, -2)
    zero = jnp.zeros_like(xe)
    # Compare with 1 and -1 explicitly: inputs whose weight is 0 are in neither sum.
    plus = jnp.where(t == 1, xe, zero).sum(axis=-1)
    minus = jnp.where(t == -1, xe, zero).sum(axis=-1)
    # The only multiplication of the layer: one per output.
    return scale * (plus - minus)
