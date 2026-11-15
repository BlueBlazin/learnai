import jax
import jax.numpy as jnp


def nf4_levels() -> jax.Array:
    """The 16 NF4 levels in ascending order, from -1 to 1, with an exact 0 at index 7.

    Returns a float32 array of shape (16,).
    """
    raise NotImplementedError


def nf4_quantize(w: jax.Array, block_size: int) -> tuple[jax.Array, jax.Array]:
    """Quantize w to NF4 codes, each block of block_size inputs scaled by its absmax.

    w: float32 array of shape (out, in); block_size divides in.
    Returns (codes, absmax): uint8 codes of shape (out, in), each the index of the
    level nearest to w / absmax, and a float32 array of shape (out, in / block_size)
    with the largest magnitude of each block (1 for an all-zero block).
    """
    raise NotImplementedError


def nf4_dequantize(codes: jax.Array, absmax: jax.Array, block_size: int) -> jax.Array:
    """Look the codes up in the table and scale each block by its absmax.

    codes: uint8 array of shape (out, in), values 0 to 15. absmax: float32 array of
    shape (out, in / block_size).
    Returns a float32 array of shape (out, in).
    """
    raise NotImplementedError
