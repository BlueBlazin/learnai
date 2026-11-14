import jax


def next_token_logits(hidden: jax.Array, w_out: jax.Array) -> jax.Array:
    # Select first, project second: only the last position predicts the next token,
    # so the (V, D) matrix is applied to B vectors instead of B * T.
    last = hidden[:, -1]  # (B, D)
    return last @ w_out.T  # (B, V); w_out is (out, in)
