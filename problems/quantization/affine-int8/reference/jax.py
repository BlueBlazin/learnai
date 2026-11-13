import jax
import jax.numpy as jnp

QMIN, QMAX = -128, 127


def quant_params(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Widen the range to include 0, so that 0.0 lands exactly on a grid point.
    lo = jnp.minimum(jnp.min(x), 0.0)
    hi = jnp.maximum(jnp.max(x), 0.0)
    # 256 codes have 255 steps between them. An all-zero tensor gets scale 1.
    scale = jnp.where(hi > lo, (hi - lo) / (QMAX - QMIN), jnp.ones_like(hi))
    # lo must map to QMIN: round(lo / scale) + zero_point = QMIN.
    zero_point = jnp.clip(QMIN - jnp.round(lo / scale), QMIN, QMAX)
    return scale, zero_point.astype(jnp.int32)


def quantize(x: jax.Array, scale: jax.Array, zero_point: jax.Array) -> jax.Array:
    q = jnp.round(x / scale) + zero_point
    # Clamp while still in float: do not rely on what an out-of-range cast does.
    return jnp.clip(q, QMIN, QMAX).astype(jnp.int8)


def dequantize(q: jax.Array, scale: jax.Array, zero_point: jax.Array) -> jax.Array:
    # Widen before subtracting: q - zero_point spans [-255, 255], which overflows int8.
    return (q.astype(jnp.int32) - zero_point).astype(jnp.float32) * scale
