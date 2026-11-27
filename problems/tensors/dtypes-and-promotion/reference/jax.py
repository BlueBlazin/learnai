import jax
import jax.numpy as jnp


def dequantize(q: jax.Array, scale: float, zero_point: int) -> jax.Array:
    # q - zero_point spans [-255, 255], which int8 cannot hold, and an int8 array
    # minus a Python int stays int8. So leave int8 before subtracting. float32 holds
    # every integer of that range exactly.
    return (q.astype(jnp.float32) - zero_point) * scale


def saturating_add_u8(a: jax.Array, b: jax.Array) -> jax.Array:
    # The sum reaches 510. Add in int16, where it fits, clamp there, and only then
    # return to uint8: after the clamp every value is in range, so the cast is exact.
    total = a.astype(jnp.int16) + b.astype(jnp.int16)
    return jnp.minimum(total, 255).astype(jnp.uint8)


def block_and_offset(pos: jax.Array, size: int) -> tuple[jax.Array, jax.Array]:
    # Floor division rounds toward negative infinity, and remainder is the one that
    # goes with it: block * size + offset == pos, with 0 <= offset < size.
    return jnp.floor_divide(pos, size), jnp.remainder(pos, size)
