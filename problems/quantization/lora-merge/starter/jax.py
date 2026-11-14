import jax
import jax.numpy as jnp


def merge(w: jax.Array, a: jax.Array, b: jax.Array, alpha: float) -> jax.Array:
    """Fold the adapter into the weight: w + (alpha / r) * b @ a.

    w: float32 (out, in). a: float32 (r, in). b: float32 (out, r). alpha: Python float.
    Returns a float32 tensor of shape (out, in).
    """
    raise NotImplementedError


def unmerge(w_merged: jax.Array, a: jax.Array, b: jax.Array, alpha: float) -> jax.Array:
    """Undo merge: recover w from merge(w, a, b, alpha), given the same adapter.

    w_merged: float32 (out, in). a: float32 (r, in). b: float32 (out, r).
    Returns a float32 tensor of shape (out, in).
    """
    raise NotImplementedError


def lora_forward(x: jax.Array, w: jax.Array, a: jax.Array, b: jax.Array, alpha: float) -> jax.Array:
    """The adapted layer without merging: x @ w.T + (alpha / r) * (x @ a.T) @ b.T.

    x: float32 (..., in), one to three dimensions. w, a, b, alpha: as in merge.
    Returns a float32 tensor of shape (..., out).
    """
    raise NotImplementedError
