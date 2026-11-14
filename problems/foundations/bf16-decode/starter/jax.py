import jax
import jax.numpy as jnp


def bf16_to_f32(bits: jax.Array) -> jax.Array:
    """Decode bfloat16 bit patterns into the float32 values they represent, exactly.

    bits: uint16 array of any shape with 1 to 3 dimensions; every pattern is valid.
    Returns a float32 array of the same shape. NaN payloads and signs are preserved.
    """
    raise NotImplementedError


def f32_to_bf16(x: jax.Array) -> jax.Array:
    """Encode float32 values as bfloat16 bit patterns, rounding to nearest, ties to even.

    x: float32 array of any shape with 1 to 3 dimensions; may hold infinities and NaN.
    Returns a uint16 array of the same shape. NaN becomes 0x7FC0, or 0xFFC0 when its
    sign bit is set.
    """
    raise NotImplementedError
