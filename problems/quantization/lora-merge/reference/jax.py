import jax


def merge(w: jax.Array, a: jax.Array, b: jax.Array, alpha: float) -> jax.Array:
    # The rank is the number of rows of a; the update b @ a has the shape of w.
    return w + (alpha / a.shape[0]) * (b @ a)


def unmerge(w_merged: jax.Array, a: jax.Array, b: jax.Array, alpha: float) -> jax.Array:
    # Subtract exactly what merge added.
    return w_merged - (alpha / a.shape[0]) * (b @ a)


def lora_forward(x: jax.Array, w: jax.Array, a: jax.Array, b: jax.Array, alpha: float) -> jax.Array:
    # Base path plus the low-rank path: project down to r dimensions with a, back up with b.
    # The (out, in) update matrix is never formed.
    return x @ w.T + (alpha / a.shape[0]) * ((x @ a.T) @ b.T)
