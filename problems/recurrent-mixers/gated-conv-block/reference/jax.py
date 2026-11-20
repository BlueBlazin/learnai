import jax
import jax.numpy as jnp


def conv_block(
    x: jax.Array,
    w_in: jax.Array,
    w_conv: jax.Array,
    w_out: jax.Array,
    state: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    steps, kernel = x.shape[0], w_conv.shape[1]
    # Three equal parts, in this order: input gate, output gate, signal.
    B, C, h = jnp.split(x @ w_in.T, 3, axis=-1)
    u = B * h  # what the convolution reads, and what the state remembers
    if state is None:
        state = jnp.zeros((kernel - 1, u.shape[1]), dtype=u.dtype)
    padded = jnp.concatenate((state, u), axis=0)  # (L - 1 + T, D)
    # Output t reads rows t .. t + L - 1 of padded: tap j meets row t + j, so the
    # last tap multiplies the current token. No flip: this is a cross-correlation.
    conv = sum(padded[j : j + steps] * w_conv[:, j] for j in range(kernel))
    y = (C * conv) @ w_out.T
    return y, padded[steps:]  # the last L - 1 rows of u, also when T < L - 1
