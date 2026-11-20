import jax
import jax.numpy as jnp


def conv_block(
    x: jax.Array,
    w_in: jax.Array,
    w_conv: jax.Array,
    w_out: jax.Array,
    state: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """A gated causal convolution over T new tokens, continuing from `state`.

    x: float32, shape (T, D), T >= 1.
    w_in: float32, shape (3 * D, D). w_out: float32, shape (D, D). Applied as x @ w.T.
    w_conv: float32, shape (D, L). One kernel of L taps per channel; the last tap
        multiplies the current token.
    state: None for a new sequence, or float32 of shape (L - 1, D): the last L - 1
        rows of u = B * h from the tokens before x.
    Returns (y, state): y float32 of shape (T, D) and the new state, float32 of
    shape (L - 1, D).
    """
    raise NotImplementedError
